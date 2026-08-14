"""bpe.py — 从零手搓 BPE 分词器（教学版）

BPE = Byte Pair Encoding，GPT-2 / GPT-3 / LLaMA 等大模型使用的分词算法。
核心思想（一句话）：
    从「字符」开始，反复把出现频率最高的相邻字符对合并成一个新符号，
    直到词表达到目标大小。结果：常见词 = 单个 token，罕见词 = 拆成子词，
    任何词都能表示，没有「未登录词（OOV）」问题。

用法：
    python code/ch01/bpe.py --demo
    python code/ch01/bpe.py --text "任意文本"
"""

from __future__ import annotations

import argparse
from collections import Counter


class BPETokenizer:
    """极简 BPE 分词器。零依赖，纯标准库，约 100 行。

    三个核心数据结构：
      char_to_id : 字符 -> id          （初始词表）
      vocab      : id  -> 文本          （完整词表，含合并出来的新符号）
      merges     : [(a, b), ...]        （合并规则，按学习顺序保存）
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.char_to_id: dict[str, int] = {}
        self.vocab: dict[int, str] = {}
        self.merges: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # 训练：从原始文本中学出合并规则
    # ------------------------------------------------------------------
    def train(self, text: str, verbose: bool = False) -> None:
        # 第 1 步：初始词表 = 文本中出现过的所有字符
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.vocab = {i: c for c, i in self.char_to_id.items()}
        ids = [self.char_to_id[c] for c in text]          # 文本转成 id 序列

        num_merges = self.vocab_size - len(self.vocab)    # 还剩多少次合并额度
        for merge_idx in range(num_merges):
            # 统计相邻 id 对的频率：zip(ids, ids[1:]) 就是所有相邻对
            pair_freqs = Counter(zip(ids, ids[1:]))
            if not pair_freqs:
                break                                     # 文本已被合并成 1 个 token
            (a, b), freq = pair_freqs.most_common(1)[0]   # 选最频繁的对

            new_id = len(self.vocab)                      # 新符号的 id = 当前词表大小
            self.merges.append((a, b))
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]  # 新符号 = 两段文本拼接
            ids = self._merge_pair(ids, (a, b), new_id)   # 把文本里所有该对替换成新 id

            if verbose:
                print(f"  merge #{merge_idx + 1:3d}: "
                      f"{self.vocab[a]!r} + {self.vocab[b]!r} -> {self.vocab[new_id]!r} "
                      f"({freq}x)")

    # ------------------------------------------------------------------
    # 编码：把新文本转成 id 序列（贪心应用所有合并规则）
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        try:
            ids = [self.char_to_id[c] for c in text]      # 先转成初始字符 id
        except KeyError as e:
            raise ValueError(
                f"未知字符 {e!r}。训练时没见过的字符无法编码。\n"
                "工程解法：用 byte-level BPE（把 UTF-8 字节而非字符放进初始词表），"
                "任何 unicode 都能编码。见本章练习。"
            ) from None
        for new_id, (a, b) in enumerate(self.merges, start=len(self.char_to_id)):
            ids = self._merge_pair(ids, (a, b), new_id)
        return ids

    # ------------------------------------------------------------------
    # 解码：id 序列还原成文本（无损，因为合并规则可逆）
    # ------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    # ------------------------------------------------------------------
    # 工具：把序列中所有相邻的 (a, b) 替换成 new_id
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """扫描一遍，遇到 (a,b) 就吞成 new_id（合并后跳过这两个位置）。"""
        out: list[int] = []
        i = 0
        a, b = pair
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2                                    # 吞掉一对
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # 信息量
    # ------------------------------------------------------------------
    @property
    def num_merges(self) -> int:
        return len(self.merges)

    def compression_ratio(self, text: str) -> float:
        """压缩率：原始字符数 / token 数。越高说明分词越「懂」这门语言。"""
        return len(text) / max(len(self.encode(text)), 1)


# ======================================================================
# 演示
# ======================================================================
def demo() -> None:
    print("=" * 64)
    print("演示 1：GPT 论文的经典例子 —— 学合并规则")
    print("=" * 64)
    text = ("low low low low low low low low "
            "lowest lowest lowest newer newer newer "
            "newer newer newer newer newest newest newest")
    tok = BPETokenizer(vocab_size=64)
    tok.train(text, verbose=True)
    print(f"\n初始词表大小：{len(tok.char_to_id)}，学到的合并规则：{tok.num_merges} 条")
    print(f"最终词表大小：{len(tok.vocab)}（目标 {tok.vocab_size}）")

    print()
    print("=" * 64)
    print("演示 2：中文 + 英文混合文本，验证无损往返")
    print("=" * 64)
    corpus = ("深度学习让机器学会了自己写代码。"
              "Build your own AI stack from scratch, "
              "tokenizer to agent. 从零手搓一整套 AI 技术栈。"
              "deep learning deep learning transformer attention")
    tok2 = BPETokenizer(vocab_size=128)
    tok2.train(corpus, verbose=True)

    ids = tok2.encode(corpus)
    back = tok2.decode(ids)
    assert back == corpus, "往返失败！encode -> decode 必须无损"
    print(f"\n✅ 往返无损验证通过：encode -> decode == 原文")
    print(f"原始字符数：{len(corpus)}，token 数：{len(ids)}，压缩率：{tok2.compression_ratio(corpus):.2f}x")
    print(f"\n前 20 个 token 的 id：{ids[:20]}")
    print(f"这些 id 代表什么：{[tok2.vocab[i] for i in ids[:20]]}")

    print()
    print("=" * 64)
    print("演示 3：看看 BPE 把常见词合成了单个 token")
    print("=" * 64)
    for wid, piece in sorted(tok2.vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]:
        print(f"  id {wid:3d}: {piece!r}（{len(piece)} 个字符）")

    print()
    print("💡 你现在已经拥有了 GPT-2 分词器的完整核心机制。")


def main() -> None:
    parser = argparse.ArgumentParser(description="从零手搓 BPE 分词器")
    parser.add_argument("--demo", action="store_true", help="跑内置演示")
    parser.add_argument("--text", type=str, default=None, help="对任意文本训练并展示")
    parser.add_argument("--vocab-size", type=int, default=128, help="目标词表大小")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.text:
        tok = BPETokenizer(vocab_size=args.vocab_size)
        print(f"训练文本（{len(args.text)} 字符），目标词表 {args.vocab_size}...")
        tok.train(args.text, verbose=True)
        ids = tok.encode(args.text)
        print(f"\ntoken 数：{len(ids)}，压缩率：{tok.compression_ratio(args.text):.2f}x")
        print(f"id 序列：{ids}")
        print(f"还原验证：{'✅ 无损' if tok.decode(ids) == args.text else '❌ 出错了'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
