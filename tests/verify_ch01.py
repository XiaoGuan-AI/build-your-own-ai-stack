"""verify_ch01.py — 第 1 章 BPE 验证。独立可跑，也兼容 run_all.py。"""
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
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch01"))
    from bpe import BPETokenizer

    text = ("low low low low low low low low lowest lowest lowest "
            "newer newer newer newer newer newer newest newest newest")
    tok = BPETokenizer(vocab_size=64)
    tok.train(text)
    check("初始词表=9", len(tok.char_to_id) == 9, f"(got {len(tok.char_to_id)})")
    check("合并规则=26（实测）", tok.num_merges == 26, f"(got {tok.num_merges})")
    check("往返无损", tok.decode(tok.encode(text)) == text)
    check("lowest 拆成 lo+west", [tok.vocab[i] for i in tok.encode("lowest")] == ["lo", "west"])

    corpus = ("深度学习让机器学会了自己写代码。Build your own AI stack from scratch, "
              "tokenizer to agent. 从零手搓一整套 AI 技术栈。")
    tok2 = BPETokenizer(vocab_size=128)
    tok2.train(corpus)
    check("中英往返无损", tok2.decode(tok2.encode(corpus)) == corpus)
    try:
        tok2.encode("🤖")
        check("未知字符抛 ValueError", False)
    except ValueError:
        check("未知字符抛 ValueError", True)
    check("_merge_pair 边界", BPETokenizer._merge_pair([1, 1, 1], (1, 1), 99) == [99, 1])
    return passed, failed


if __name__ == "__main__":
    p, f = run()
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
