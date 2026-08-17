"""verify_ch02.py — 第 2 章 Transformer 验证（含 KV Cache 改造回归）。"""
from __future__ import annotations

import pathlib
import sys

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name} {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


def run(ckpt: str | None = None, proj: str | None = None):
    global passed, failed
    passed = failed = 0
    proj = proj or str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch02"))
    import torch
    import torch.nn.functional as F
    from transformer import MiniTransformer

    torch.manual_seed(42)
    V = 65
    model = MiniTransformer(vocab_size=V, d_model=64, n_heads=4, n_layers=2, max_len=128)
    x = torch.randint(0, V, (2, 16))
    logits = model(x)
    check("forward 形状 (B,T,V)", logits.shape == (2, 16, V), f"(got {logits.shape})")

    # 因果性（掩码语义）
    xa, xb = x.clone(), x.clone()
    xb[:, 5] = (xb[:, 5] + 1) % V
    check("改未来不影响过去", torch.equal(model(xa)[:, :5], model(xb)[:, :5]))

    # 注意力权重
    attn = model.blocks[0].attn
    B, T = x.shape
    h = model.token_embedding(x)
    q, k, v = attn.qkv(h).split(attn.d_model, dim=2)
    q = q.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / (attn.head_dim ** 0.5)
    scores = scores.masked_fill(attn.mask[:, :, :T, :T] == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    check("注意力行和=1", torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1))))
    check("掩码位置权重=0", torch.allclose(weights[..., 1, 2], torch.zeros_like(weights[..., 1, 2])))

    # 梯度
    loss = F.cross_entropy(logits.view(-1, V), x.view(-1))
    loss.backward()
    check("梯度流通", sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None) > 0)

    # 生成 + KV Cache 一致性
    out = model.generate(torch.tensor([[1]]), max_new_tokens=40, temperature=1.0)
    check("generate 形状", out.shape == (1, 41), f"(got {out.shape})")
    ids = torch.tensor([[1, 2, 3]])
    torch.manual_seed(3)
    oc = model.generate(ids, max_new_tokens=20, temperature=0.9, top_k=40, use_cache=True)
    torch.manual_seed(3)
    of = model.generate(ids, max_new_tokens=20, temperature=0.9, top_k=40, use_cache=False)
    check("KV Cache 一致性", torch.equal(oc, of))
    # 复查#4 回归：generate 入口参数校验
    try:
        model.generate(torch.tensor([[1]]), max_new_tokens=3, temperature=0)
        check("temperature=0 报错", False)
    except ValueError:
        check("temperature=0 报错", True)
    try:
        model.generate(torch.tensor([[1]]), max_new_tokens=3, top_k=0)
        check("top_k=0 报错", False)
    except ValueError:
        check("top_k=0 报错", True)
    try:
        model.generate(torch.tensor([[1]]), max_new_tokens=3, top_p=1.5)
        check("top_p 越界报错", False)
    except ValueError:
        check("top_p 越界报错", True)

    # P2 回归：d_model 奇数构造期报错（正弦位置编码需要偶数）
    try:
        MiniTransformer(vocab_size=65, d_model=33, n_heads=1, n_layers=1)
        check("d_model 奇数报错", False)
    except AssertionError:
        check("d_model 奇数报错", True)
    # P2 回归：超长回退 cache==no-cache 一致性（prompt+new 超过 max_len）
    long = torch.randint(1, V, (1, 30))
    torch.manual_seed(7)
    oc2 = model.generate(long, max_new_tokens=120, temperature=0.9, top_k=40, use_cache=True)
    torch.manual_seed(7)
    of2 = model.generate(long, max_new_tokens=120, temperature=0.9, top_k=40, use_cache=False)
    check("超长回退一致性", torch.equal(oc2, of2))

    # 审查 S3 回归：max_len>1024 前向不再崩溃（mask 与 max_len 绑定）
    big = MiniTransformer(vocab_size=65, d_model=32, n_heads=4, n_layers=1, max_len=2048)
    out_big = big(torch.randint(0, 65, (1, 1500)))
    check("max_len=2048 前向可跑", out_big.shape == (1, 1500, 65), f"(got {out_big.shape})")
    return passed, failed


if __name__ == "__main__":
    p, f = run()
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
