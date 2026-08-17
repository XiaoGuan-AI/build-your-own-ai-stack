"""rag.py — 第 5 章：RAG（Retrieval-Augmented Generation）教学版，约 200 行

RAG = 先检索相关文档，再把文档拼进 prompt，最后让模型基于文档生成。
解决 LLM 的「知识截止」和「幻觉」：不重新训练，给模型开卷考试。

本教学版零外部依赖（不装 FAISS / sentence-transformers）：
  - 向量化：字符 n-gram 计数（bigram+trigram）→ 归一化向量
  - 检索：余弦相似度 top-k
  - 生成：复用第 2/4 章手搓的模型
真实系统只是把「n-gram 向量」换成「预训练 embedding 模型」、
把「线性扫描」换成「向量数据库」，原理完全一样。

用法：
    .venv/Scripts/python.exe code/ch03/train.py --demo          # 先有模型
    .venv/Scripts/python.exe code/ch05/rag.py --query "什么是梯度下降"
    .venv/Scripts/python.exe code/ch05/rag.py --query "KV Cache 是什么" --compare   # 对比无 RAG
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
from transformer import MiniTransformer  # noqa: E402
from train import BUILTIN_CORPUS, CharTokenizer  # noqa: E402
from inference import load_model  # noqa: E402


# ======================================================================
# 内置知识库（教学用：几段概念科普，演示「开卷考试」）
# ======================================================================
KNOWLEDGE_BASE = {
    "张量": "张量是带形状的多维数组。标量是零维张量，向量是一维张量，矩阵是二维张量。"
            "深度学习中的所有数据都以张量形式存在，模型训练就是不断调整张量里的数值。",
    "梯度下降": "梯度下降是训练神经网络的核心算法。先计算损失函数对每个参数的梯度，"
               "然后沿着负梯度方向更新参数，反复迭代直到损失收敛。学习率决定每次迈多大步。",
    "反向传播": "反向传播沿着计算图从输出向输入逐层分配误差，用链式法则求出每个参数的梯度。"
               "它让训练数百万参数的模型成为可能。",
    "Transformer": "Transformer 是当前大语言模型的基础架构，核心是注意力机制："
                   "每个 token 直接观察序列中所有其他 token，决定该关注谁。"
                   "相比 RNN，它可并行计算且没有长距离遗忘问题。",
    "注意力机制": "注意力机制用查询向量 Query 与键向量 Key 计算相关性，"
                 "经过 softmax 变成权重，再对值向量 Value 加权求和。"
                 "除以根号 d 是为了防止 softmax 饱和导致梯度消失。",
    "BPE分词": "BPE 字节对编码是大模型使用的分词算法，从字符开始反复合并出现频率最高的相邻对，"
               "直到词表达到目标大小。常见词成为单个 token，罕见词拆成子词，没有未登录词问题。",
    "KV缓存": "KV Cache 是推理加速技术，把每层注意力已经算过的 Key 和 Value 缓存起来，"
              "生成新 token 时只计算新位置的 QKV，避免重复计算整个前缀，"
              "把每步生成成本从 O(序列长度) 降到 O(1)。",
    "采样策略": "采样策略决定模型如何从概率分布中选 token。贪心解码每步取概率最大的 token，"
                "容易陷入复读；temperature 控制分布的尖锐程度；top-k 只保留前 k 个候选；"
                "top-p 按累计概率动态截断，是当前主流模型的默认选择。",
}


# ======================================================================
# 5.1 文本向量化：字符 n-gram 计数（穷人版 embedding）
# ======================================================================
def ngram_vector(text: str, n: int = 2) -> Counter:
    """统计文本中所有长度为 n 的连续字符子串（n-gram）出现次数。"""
    text = text.replace(" ", "").replace("\n", "")     # 忽略空白，聚焦内容
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def vectorize(text: str) -> dict:
    """bigram + trigram 计数拼接，再按 L2 范数归一化（长度无关的向量）。"""
    vec = ngram_vector(text, 2)
    vec.update(ngram_vector(text, 3))
    norm = math.sqrt(sum(c * c for c in vec.values()))
    return {k: v / norm for k, v in vec.items()} if norm > 0 else {}


def cosine_similarity(a: dict, b: dict) -> float:
    """两个稀疏向量的余弦相似度（点积，因为都已归一化）。"""
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(v * large.get(k, 0.0) for k, v in small.items())


# ======================================================================
# 5.2 语料库与检索
# ======================================================================
class CorpusStore:
    """文档库：切块 → 向量化 → 余弦检索 top-k。"""

    def __init__(self, knowledge_base: dict):
        self.chunks = [(title, content) for title, content in knowledge_base.items()]
        self.vectors = [vectorize(content) for _, content in self.chunks]

    def retrieve(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        """返回 [(标题, 内容, 相似度)]，按相似度降序。"""
        qv = vectorize(query)
        scored = [(cosine_similarity(qv, v), i) for i, v in enumerate(self.vectors)]
        scored.sort(reverse=True)
        return [(self.chunks[i][0], self.chunks[i][1], s) for s, i in scored[:k]]


# ======================================================================
# 5.3 生成：检索 → 拼 prompt → 让模型基于上下文回答
# ======================================================================
def build_prompt(query: str, contexts: list[tuple[str, str, float]]) -> str:
    """把检索到的知识 + 问题拼成一个 prompt（开卷考试的试卷）。

    标记用 ASCII（Q:/A:/等）：字符级词表没有全角符号，全角会触发 OOV
    （这就是第 1 章 BPE 的卖点——真实系统不会遇到这个问题）。
    """
    parts = ["Background knowledge:"]
    for title, content, score in contexts:
        parts.append(f"  [{title}] {content}")
    parts.append(f"\nQ: {query}")
    parts.append("\nA: Answer based on the background knowledge above:")
    return "\n".join(parts)


def sanitize(text: str, tokenizer: CharTokenizer) -> str:
    """把词表外字符替换为词表内字符（字符级分词器的 OOV 兜底；BPE 无此问题）。

    复查#6：兜底字符必须取词表内的（空格可能不在无空格语料的词表里）。
    """
    fallback = " " if " " in tokenizer.stoi else next(iter(tokenizer.stoi))
    return "".join(c if c in tokenizer.stoi else fallback for c in text)


def generate_with_context(model: MiniTransformer, tokenizer: CharTokenizer,
                          prompt: str, max_new_tokens: int,
                          temperature: float, top_k: int, device: str) -> str:
    # 模型只能看到词表内的字符；prompt 超长时截断到模型窗口内
    prompt = sanitize(prompt, tokenizer)
    prompt = prompt[: model.max_len - max_new_tokens - 2]
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="第 5 章：RAG 检索增强生成")
    parser.add_argument("--ckpt", default="checkpoints/ch03-demo.pt", help="模型 checkpoint")
    parser.add_argument("--query", default="什么是梯度下降", help="问题")
    parser.add_argument("--k", type=int, default=3, help="检索条数")
    parser.add_argument("--max-new-tokens", type=int, default=60, help="生成长度")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--compare", action="store_true", help="对比无 RAG 的直接生成")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tokenizer = CharTokenizer(BUILTIN_CORPUS)

    store = CorpusStore(KNOWLEDGE_BASE)
    print(f"\n===== 检索（query={args.query!r}，top-{args.k}）=====")
    hits = store.retrieve(args.query, args.k)
    for title, content, score in hits:
        print(f"  [{title}] 相似度 {score:.3f}：{content[:38]}…")

    print(f"\n===== RAG 生成（基于检索到的知识）=====")
    prompt = build_prompt(args.query, hits)
    print(f"【prompt 前 80 字】{prompt[:80]}…")
    answer = generate_with_context(model, tokenizer, prompt,
                                   args.max_new_tokens, args.temperature, args.top_k, device)
    print(f"【回答】{answer}")

    if args.compare:
        print(f"\n===== 无 RAG 直接生成（对照）=====")
        plain = generate_with_context(model, tokenizer, f"【问题】{args.query}\n【回答】",
                                      args.max_new_tokens, args.temperature, args.top_k, device)
        print(f"【回答】{plain}")
        print("\n对比：无 RAG 时模型只能凭训练语料瞎编（小模型=乱码）；"
              "有 RAG 时 prompt 里给了相关内容，模型会引用背景知识——这就是开卷考试。")


if __name__ == "__main__":
    main()
