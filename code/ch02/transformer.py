"""transformer.py — 从零手搓 Transformer（教学版，约 250 行）

只用 PyTorch 的「积木」（Linear / LayerNorm / softmax），注意力机制全部手搓。
这就是 GPT 系列 decoder-only 架构的核心前向传播，第 3 章你将用它训练出真模型。

用法：
    .venv/Scripts/python.exe code/ch02/transformer.py --demo
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 2.2~2.4 缩放点积注意力 + 多头 + 因果掩码
# ======================================================================
class CausalSelfAttention(nn.Module):
    """多头因果自注意力。

    一条公式串起全部：
        attention = softmax( (Q·Kᵀ)/√d + 掩码 ) · V
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads          # 每个头的维度

        # QKV 投影合成一个 Linear（参数效率：一个矩阵顶三个）
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)   # 拼头后的输出投影
        # 因果掩码：上三角为 False（未来位置），登记为 buffer 随模型移动
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 1024, 1024)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                       # B=batch, T=序列长, C=d_model
        q, k, v = self.qkv(x).split(self.d_model, dim=2)   # 一个 Linear 拆成 Q/K/V

        # (B, T, C) -> (B, n_heads, T, head_dim)：把 C 拆成 n_heads 个头
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力：q·kᵀ / √head_dim  ← 为什么要除以 √d：softmax 饱和
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
        # 因果掩码：把未来位置(上三角)设成 -inf，softmax 后权重为 0
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)            # 每行权重归一化（和为 1）

        y = att @ v                             # 加权求和：注意力真正「带走的」信息
        # 拼回多头：(B, n_heads, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)                     # 输出投影混合各头信息


# ======================================================================
# 2.7 前馈网络（FFN）：每个 token 独立的非线性思考
# ======================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),    # 先放大 4 倍
            nn.GELU(),                          # 非线性
            nn.Linear(4 * d_model, d_model),    # 压回原维度
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# 2.6~2.7 Block：残差 + LayerNorm + 注意力 + 残差 + LayerNorm + FFN
# ======================================================================
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))          # 残差：注意力开完会，结果加回原路
        x = x + self.ffn(self.ln2(x))           # 残差：FFN 消化完，再加回原路
        return x


# ======================================================================
# 2.5~2.8 MiniTransformer：embedding + 位置编码 + N 个 Block + 输出头
# ======================================================================
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, max_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)   # 输出词表分数

        self._init_sinusoidal_positions(max_len, d_model)

    def _init_sinusoidal_positions(self, max_len: int, d_model: int):
        """2.5 正弦位置编码：给每个位置一个独一无二的「指纹」"""
        pos = torch.arange(max_len).unsqueeze(1)                    # (max_len, 1)
        i = torch.arange(0, d_model, 2)                             # 偶数维度
        angles = pos / (10000 ** (i / d_model))                     # 频率衰减
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)                             # 偶数维 sin
        pe[:, 1::2] = torch.cos(angles)                             # 奇数维 cos
        self.register_buffer("pos_encoding", pe.unsqueeze(0))       # (1, max_len, d)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) token ids -> logits: (B, T, vocab_size)"""
        B, T = idx.shape
        assert T <= self.max_len, f"序列超长：{T} > {self.max_len}"
        x = self.token_embedding(idx)                               # (B, T, d)
        x = x + self.pos_encoding[:, :T, :]                         # 加位置指纹
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)                                      # (B, T, vocab)

    # ------------------------------------------------------------------
    # 2.4/2.9 推理：逐个 token 采样（预测 -> 拼接 -> 再预测）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0) -> torch.Tensor:
        """给定提示 (B, T)，逐 token 生成 max_new_tokens 个，返回完整序列"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]                       # 只看最近 max_len
            logits = self(idx_cond)                                 # (B, T, vocab)
            logits = logits[:, -1, :] / temperature                 # 只看最后位置 + 温度
            probs = F.softmax(logits, dim=-1)                       # 概率分布
            next_id = torch.multinomial(probs, num_samples=1)       # 按概率采样
            idx = torch.cat([idx, next_id], dim=1)                  # 拼到序列尾巴
        return idx


# ======================================================================
# 演示与自检
# ======================================================================
def demo() -> None:
    torch.manual_seed(42)
    vocab_size = 65   # 字符级词表（经典 tiny shakespeare 风格）
    model = MiniTransformer(vocab_size=vocab_size, d_model=64, n_heads=4,
                            n_layers=2, max_len=128)
    print("MiniTransformer 参数量:", sum(p.numel() for p in model.parameters()), "参数")

    print()
    print("=" * 64)
    print("自检 1：前向形状与因果掩码")
    print("=" * 64)
    x = torch.randint(0, vocab_size, (2, 16))       # 2 条样本，每条 16 个 token
    logits = model(x)
    print(f"输入 (B,T)={tuple(x.shape)} -> logits {tuple(logits.shape)} ✅ 形状正确")

    # 验证因果性：把第 5 个 token 改成别的，位置 3 的输出必须完全不变
    x_a = x.clone()
    x_b = x.clone()
    x_b[:, 5] = (x_b[:, 5] + 1) % vocab_size
    out_a, out_b = model(x_a), model(x_b)
    assert torch.equal(out_a[:, :5, :], out_b[:, :5, :]), "因果性被破坏！"
    print("因果性验证：改未来 token，过去位置的输出纹丝不动 ✅")

    # 验证注意力行和为 1（softmax 性质）
    attn = model.blocks[0].attn
    B, T = x.shape
    h = model.token_embedding(x)                    # ids -> 向量（(2,16,64)）
    q, k, v = attn.qkv(h).split(attn.d_model, dim=2)
    q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    print("注意力权重行和为 1 ✅（softmax 归一化正确）")

    print()
    print("=" * 64)
    print("自检 2：反向传播能跑（梯度存在）")
    print("=" * 64)
    loss = F.cross_entropy(logits.view(-1, vocab_size), x.view(-1))
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"loss = {loss.item():.4f}，全参数梯度范数和 = {grad_norm:.3f} ✅ 梯度流通正常")

    print()
    print("=" * 64)
    print("自检 3：随机权重生成（未训练，输出是乱码但结构已跑通）")
    print("=" * 64)
    prompt = torch.tensor([[1]])                   # 从 id=1 开始
    generated = model.generate(prompt, max_new_tokens=80, temperature=1.0)
    print("生成 80 个 token 完成，形状:", tuple(generated.shape))
    print("💡 这 80 步的循环（预测→采样→拼接）就是所有大模型生成回答的方式。")
    print("   第 3 章训练后，同样的 generate() 会输出真正的句子。")


def main() -> None:
    parser = argparse.ArgumentParser(description="从零手搓 Transformer（第 2 章）")
    parser.add_argument("--demo", action="store_true", help="跑自检与演示")
    parser.add_argument("--d-model", type=int, default=64, help="模型宽度")
    parser.add_argument("--n-heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer 层数")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
