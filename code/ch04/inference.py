"""inference.py — 第 4 章：推理引擎（教学版，约 200 行）

把第 3 章训练好的模型变成真正能用的服务。三件武器：
  1. KV Cache：把每步生成从 O(序列长) 降到 O(1) 的前向成本
  2. 采样策略：temperature / top-k / top-p，控制模型「说什么话」
  3. 服务循环：加载 checkpoint -> 输入提示 -> 生成 -> 返回

用法：
    # 先训练一个模型（或复用已有 checkpoint）
    .venv/Scripts/python.exe code/ch03/train.py --demo
    # 用推理引擎生成一段（默认加载 checkpoints/ch03-demo.pt）
    .venv/Scripts/python.exe code/ch04/inference.py --prompt "床前明月"
    # 对比 KV Cache 加速比
    .venv/Scripts/python.exe code/ch04/inference.py --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
from transformer import MiniTransformer  # noqa: E402
from train import BUILTIN_CORPUS, CharTokenizer  # noqa: E402


# ======================================================================
# 4.1 模型加载：checkpoint 里现在存了 cfg（配置），无需手工传参
# ======================================================================
def load_model(ckpt_path: str, device: str = "cpu"):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert "cfg" in state, (
        "旧版 checkpoint 没有 cfg。请重新训练：.venv/Scripts/python.exe code/ch03/train.py --demo"
    )
    model = MiniTransformer(**state["cfg"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"已加载模型：{state['cfg']} | 训练到第 {state['step']} 步，loss {state['loss']:.4f}")
    return model, state["cfg"]


# ======================================================================
# 4.2 采样策略：temperature -> top-k -> top-p -> 概率采样
# ======================================================================
def sample_token(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """给定最后位置的 logits (B, vocab)，按策略采一个 token id（形状 (B, 1)）。

    - temperature: <1 更保守（分布更尖），>1 更发散（分布更平）
    - top_k: 只保留概率最大的 k 个候选
    - top_p (nucleus): 按概率降序累计，裁掉累计概率超过 p 的尾部
    """
    if temperature <= 0:                        # 审查 M8 修复：≤0 会产生 NaN 采样
        raise ValueError("temperature 必须 > 0")
    logits = logits / temperature

    if top_k is not None:                       # 硬性候选数上限
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
    if top_p is not None:                       # 动态候选数：保留累计概率 p 的头部
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs_sorted = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs_sorted.cumsum(dim=-1)
        remove = cumprobs - probs_sorted > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ======================================================================
# 4.3 生成（包装 ch02 的 generate，传入采样参数）
# ======================================================================
def generate(model: MiniTransformer, prompt_ids: torch.Tensor,
             max_new_tokens: int = 100, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             use_cache: bool = True) -> torch.Tensor:
    return model.generate(prompt_ids, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p,
                          use_cache=use_cache)


# ======================================================================
# 4.4 基准测试：KV Cache 到底快多少
# ======================================================================
def benchmark(model: MiniTransformer, tokenizer: CharTokenizer,
              prompt: str, max_new_tokens: int = 200, device: str = "cpu") -> None:
    # 教学版 KV Cache 的正确性保证在序列 ≤ max_len 内，超出会回退完整前向。
    # 基准测试限制生成长度，保证 Cache 全程生效（加速比才是「纯加速」）。
    max_new_tokens = min(max_new_tokens, model.max_len - len(prompt) - 5)
    print(f"\n===== KV Cache 基准（prompt={prompt!r}，生成 {max_new_tokens} token）=====")
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    prompt_len = len(prompt)

    for use_cache in (True, False):
        torch.manual_seed(42)                       # 保证两边生成同一序列
        t0 = time.perf_counter()
        out = generate(model, prompt_ids, max_new_tokens=max_new_tokens,
                       temperature=0.8, top_k=40, use_cache=use_cache)
        dt = time.perf_counter() - t0
        speed = max_new_tokens / dt
        label = "KV Cache ✅" if use_cache else "无 Cache  "
        print(f"{label}：{dt:.2f}s（{speed:.1f} token/s），生成长度 {out.size(1) - prompt_len}")
        if use_cache:
            cached_out = out
    # 一致性验证（同 seed 同参数，两路输出必须逐 token 相同）
    assert torch.equal(cached_out, out), "❌ KV Cache 与非 Cache 输出不一致！"
    print(f"✅ 一致性验证：Cache 与非 Cache 生成序列完全一致")


# ======================================================================
# 4.5 服务循环：一次生成 or 交互式聊天
# ======================================================================
def run_once(model: MiniTransformer, tokenizer: CharTokenizer, prompt: str,
             max_new_tokens: int, temperature: float, top_k: int | None,
             top_p: float | None, device: str) -> None:
    # 审查 M4 修复：字符级词表 OOV 兜底（真实系统用 BPE 无此问题）
    prompt = "".join(c if c in tokenizer.stoi else " " for c in prompt)
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = generate(model, prompt_ids, max_new_tokens=max_new_tokens,
                   temperature=temperature, top_k=top_k, top_p=top_p)
    text = tokenizer.decode(out[0].tolist())
    print(f"\n【输入】{prompt}")
    print(f"【生成】{text}")
    n = max_new_tokens
    print(f"\n（生成长度 {len(out[0]) - len(prompt_ids[0])} token，"
          f"参数：temperature={temperature}, top_k={top_k}, top_p={top_p}）")


def run_interactive(model: MiniTransformer, tokenizer: CharTokenizer,
                    max_new_tokens: int, temperature: float, top_k: int | None,
                    top_p: float | None, device: str) -> None:
    print("\n交互模式：输入提示回车生成（Ctrl+C / 输入 exit 退出）")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue
        run_once(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, device)


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="第 4 章：推理引擎")
    parser.add_argument("--ckpt", type=str, default="checkpoints/ch03-demo.pt", help="模型 checkpoint")
    parser.add_argument("--prompt", type=str, default="床前明月光", help="生成提示")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="生成长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-k", type=int, default=40, help="top-k 采样（None 关闭）")
    parser.add_argument("--top-p", type=float, default=None, help="top-p 采样（None 关闭）")
    parser.add_argument("--benchmark", action="store_true", help="跑 KV Cache 基准")
    parser.add_argument("--interactive", action="store_true", help="交互式聊天")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tokenizer = CharTokenizer(BUILTIN_CORPUS)   # 词表必须与训练语料一致（见文档 4.1）

    if args.benchmark:
        benchmark(model, tokenizer, args.prompt, args.max_new_tokens, device)
    elif args.interactive:
        run_interactive(model, tokenizer, args.max_new_tokens,
                        args.temperature, args.top_k, args.top_p, device)
    else:
        run_once(model, tokenizer, args.prompt, args.max_new_tokens,
                 args.temperature, args.top_k, args.top_p, device)


if __name__ == "__main__":
    main()
