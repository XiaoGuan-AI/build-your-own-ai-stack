"""verify_ch05.py — 第 5 章 RAG 验证（检索相关性核心）。"""
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
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch05"))
    import rag

    store = rag.CorpusStore(rag.KNOWLEDGE_BASE)
    for q, expect in [("什么是梯度下降", "梯度下降"), ("KV Cache 如何加速", "KV缓存"),
                      ("BPE 怎么切词", "BPE分词"), ("注意力怎么算相关性", "注意力机制")]:
        hit = store.retrieve(q, k=1)[0]
        check(f"「{q[:6]}」→ 命中 {expect}", hit[0] == expect, f"(got {hit[0]}, {hit[2]:.3f})")

    v1 = rag.vectorize("梯度下降是训练神经网络的核心算法")
    v2 = rag.vectorize("梯度下降的核心思想是沿着负梯度方向更新参数")
    v3 = rag.vectorize("今天天气很好适合出门散步")
    check("同主题相似度 > 异主题", rag.cosine_similarity(v1, v2) > rag.cosine_similarity(v1, v3))
    check("自相似度 = 1", abs(rag.cosine_similarity(v1, v1) - 1.0) < 1e-6)
    san = rag.sanitize("什么是【梯度】下降", rag.CharTokenizer("梯度下降"))
    check("sanitize 过滤 OOV", "【" not in san and "梯度" in san and "下降" in san)

    hits = store.retrieve("什么是梯度下降", k=3)
    p = rag.build_prompt("什么是梯度下降", hits)
    check("prompt 构造", "Background knowledge:" in p and "Q: " in p and "梯度下降是训练神经网络" in p)
    return passed, failed


if __name__ == "__main__":
    p, f = run()
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
