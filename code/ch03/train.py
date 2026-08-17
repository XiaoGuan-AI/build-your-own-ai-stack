"""train.py — 第 3 章：从零预训练一个小 LLM（教学版，约 200 行）

把第 2 章手搓的 MiniTransformer 真正训起来。
训练 = 反复做一件事：给模型看文本片段，让它预测下一个 token，猜错了就调整参数。

用法：
    .venv/Scripts/python.exe code/ch03/train.py --demo
        # CPU 快速演示：内置唐诗语料，约 10 秒，肉眼可见 loss 下降
    .venv/Scripts/python.exe code/ch03/train.py --data data/tiny_shakespeare.txt --epochs 5000
        # 认真跑：自定义语料（第 3 章文档 3.9 有进阶玩法）
    .venv/Scripts/python.exe code/ch03/train.py --resume
        # 从 checkpoint 继续训练
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# 复用第 2 章手搓的 Transformer（这就是「积木复用」）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
from transformer import MiniTransformer  # noqa: E402

# ----------------------------------------------------------------------
# 内置语料：几首唐诗 + 一段白话。字符级训练，文本结构让学习效果肉眼可见
# ----------------------------------------------------------------------
BUILTIN_CORPUS = """床前明月光疑是地上霜举头望明月低头思故乡
春眠不觉晓处处闻啼鸟夜来风雨声花落知多少
白日依山尽黄河入海流欲穷千里目更上一层楼
千山鸟飞绝万径人踪灭孤舟蓑笠翁独钓寒江雪
锄禾日当午汗滴禾下土谁知盘中餐粒粒皆辛苦
AI 工程师的学习之路：先手搓分词器，再手搓 Transformer，
然后亲手把模型训练出来。Build your own AI stack from scratch.
Tokenizer to agent. 从零手搓一整套 AI 技术栈。
"""


# ----------------------------------------------------------------------
# 字符级词表（第 3 章用最简方案，专注训练；第 1 章的 BPE 是工程升级）
# ----------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


# ----------------------------------------------------------------------
# 数据采样：从语料里随机切块（每步的起点随机）
# 为什么随机起点：相邻字符高度相关，随机切块 ≈ 数据增强 + 天然正则化
# ----------------------------------------------------------------------
def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------
# 训练主循环
# ----------------------------------------------------------------------
def train(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 数据 ----
    if args.data:
        text = Path(args.data).read_text(encoding="utf-8")
        print(f"语料：{args.data}（{len(text):,} 字符）")
    else:
        text = BUILTIN_CORPUS
        print(f"语料：内置唐诗+白话（{len(text):,} 字符）")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # ---- 模型（第 2 章的积木；--demo 用小配置，认真跑用大配置）----
    cfg = dict(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
               n_heads=args.n_heads, n_layers=args.n_layers, max_len=args.block_size)

    # ---- checkpoint 恢复（第 4 章起：配置也存进 checkpoint，resume 无需重传）----
    ckpt_path = Path(args.ckpt)
    start_step = 0
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "cfg" in state:
            cfg = state["cfg"]
            args.block_size = cfg["max_len"]    # 复查#9：block-size 以训练时为准
            print(f"已从 checkpoint 恢复配置：{cfg}")
        model = MiniTransformer(**cfg)
        model.load_state_dict(state["model"], strict=False)   # 复查#1：兼容旧 ckpt mask 键
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"已从 checkpoint 恢复（第 {start_step} 步，loss {state['loss']:.4f}）")
    else:
        model = MiniTransformer(**cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型：{args.d_model}d/{args.n_heads}头/{args.n_layers}层 | "
          f"{n_params:,} 参数 | 设备：{device} | 词表：{tokenizer.vocab_size}")

    # ---- 优化器（AdamW：Adam + 权重衰减，大模型标配）----
    best_loss = float("inf")

    print(f"开始训练 {args.epochs} 步（block={args.block_size}, batch={args.batch_size}, lr={args.lr}）\n")
    t0 = time.time()

    for step in range(start_step, args.epochs):
        # 前向 + 损失：对整段序列算交叉熵（等价于逐 token 预测的平均难度）
        xb, yb = get_batch(data, args.batch_size, args.block_size, device)
        logits = model(xb)                                  # (B, T, vocab)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        perplexity = math.exp(min(loss.item(), 20))         # 困惑度 = 平均候选数

        # 反向 + 更新（梯度裁剪防爆炸，见文档 3.7）
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---- 观察点：每 100 步打印 + 生成一段 ----
        if step % 100 == 0 or step == args.epochs - 1:
            elapsed = time.time() - t0
            speed = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            print(f"step {step:5d} | loss {loss.item():.4f} | ppl {perplexity:7.1f} | "
                  f"{speed:.1f} 步/秒 | {elapsed:.0f}s", end="")
            if loss.item() < best_loss:
                best_loss = loss.item()
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)   # 验收 P0-1：先建目录
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "loss": loss.item(), "cfg": cfg}, ckpt_path)
                print("  💾 checkpoint")
            else:
                print()

            if step % 500 == 0 or step == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    seed = torch.tensor([tokenizer.stoi["床"]], device=device).unsqueeze(0)
                    sample = model.generate(seed, max_new_tokens=40, temperature=0.8)
                    print("  生成样本：" + tokenizer.decode(sample[0].tolist()).replace("\n", "⏎"))
                model.train()

    print(f"\n✅ 训练完成。最佳 loss {best_loss:.4f}，checkpoint 保存在 {ckpt_path}")
    print("下一步：跑 --resume 继续训练，或用 --demo 看完整效果。")


def main() -> None:
    parser = argparse.ArgumentParser(description="第 3 章：从零预训练一个小 LLM")
    parser.add_argument("--demo", action="store_true", help="CPU 快速演示（内置语料，1~2 分钟）")
    parser.add_argument("--data", type=str, default=None, help="自定义语料 txt（可选）")
    parser.add_argument("--epochs", type=int, default=2000, help="训练步数")
    parser.add_argument("--batch-size", type=int, default=16, help="每步样本数")
    parser.add_argument("--block-size", type=int, default=64, help="每样本序列长度")
    parser.add_argument("--d-model", type=int, default=96, help="模型宽度")
    parser.add_argument("--n-heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--n-layers", type=int, default=3, help="Transformer 层数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ckpt", type=str, default="checkpoints/ch03-demo.pt", help="checkpoint 路径")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续")
    args = parser.parse_args()

    if args.demo:  # 快速配置：小模型 + 少步数，CPU 约 1~2 分钟
        args.epochs = 500
        args.d_model = 64
        args.n_heads = 4
        args.n_layers = 2
        args.block_size = 64
        args.batch_size = 16
    train(args)


if __name__ == "__main__":
    main()
