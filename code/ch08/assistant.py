"""assistant.py — 第 8 章：大作业——迷你 AI 助手（约 230 行）

把前 7 章手搓的积木拼成一个能对话、能查知识、能算数的迷你助手：
  ch02 Transformer + ch03 预训练 + ch04 推理引擎（KV Cache/采样）
  + ch05 RAG（知识库检索）+ ch06 工具（calc/now）+ ch07 协议（schema 校验）

架构（问题路由 → 执行 → 回答）：
  用户输入
    ├─ 命令？(/help /tools /quit)   → 直接处理
    ├─ 数学题？（含数字+运算符）    → calc 工具 → 回答
    ├─ 知识问题？（命中知识库）     → RAG 检索 → 基于上下文生成
    └─ 闲聊/其他                    → 直接生成（小模型尽力而为）

用法：
    .venv/Scripts/python.exe code/ch03/train.py --demo    # 先有模型
    .venv/Scripts/python.exe code/ch08/assistant.py       # 交互模式
    .venv/Scripts/python.exe code/ch08/assistant.py --ask "什么是梯度下降"
    .venv/Scripts/python.exe code/ch08/assistant.py --ask "2+3*4 等于多少"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch05"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch06"))
from transformer import MiniTransformer  # noqa: E402
from train import BUILTIN_CORPUS, CharTokenizer  # noqa: E402
from inference import load_model  # noqa: E402
from rag import KNOWLEDGE_BASE, CorpusStore, sanitize  # noqa: E402
from agent import ToolRegistry  # noqa: E402

# 数学题检测：包含数字并且包含运算符（或两个数字）
MATH_RE = re.compile(r"[\d．.０-９]")
OP_RE = re.compile(r"[+\-*/×÷^%加加减减乘除]")


class MiniAssistant:
    """迷你 AI 助手：路由 + 工具 + RAG + 生成。"""

    def __init__(self, ckpt: str = "checkpoints/ch03-demo.pt", device: str = "cpu"):
        self.model, self.cfg = load_model(ckpt, device)
        self.tokenizer = CharTokenizer(BUILTIN_CORPUS)
        self.tools = ToolRegistry()               # ch06 工具（calc/now/search）
        self.store = CorpusStore(KNOWLEDGE_BASE)  # ch05 知识库
        self.device = device

    # ------------------------------------------------------------------
    def answer(self, question: str) -> str:
        """主入口：路由到合适的处理路径。"""
        q = question.strip()

        if q in ("/help",):
            return ("可用命令：/help /tools /quit\n"
                    "直接提问即可：数学题（如 2+3*4）、概念问题（如 什么是梯度下降）、随便聊聊。")
        if q in ("/tools",):
            return "工具清单：" + "、".join(self.tools.tools) + \
                   "（calc 计算 / search 知识检索 / now 时间）"
        if q in ("/quit", "/exit"):
            return "__QUIT__"

        # 路由 1：数学题 → calc 工具
        if MATH_RE.search(q) and OP_RE.search(q):
            expr = q.replace("等于多少", "").replace("是多少", "").replace("？", "").replace("?", "")
            result = self.tools.call("calc", expr)
            if not result.startswith("错误"):
                return f"{expr} = {result}"
            return f"（计算失败：{result}）让我直接回答：{self._generate(q)}"

        # 路由 2：知识问题 → RAG（检索 + 基于上下文生成）
        hits = self.store.retrieve(q, k=2)
        if hits and hits[0][2] > 0.02:            # 检索命中且相似度够
            contexts = "；".join(f"{t}：{c[:60]}" for t, c, _ in hits)
            return (f"📚 知识库检索（相关度 {hits[0][2]:.2f}）：\n"
                    f"  [{hits[0][0]}] {hits[0][1]}\n"
                    f"🤖 基于上下文的回答：{self._generate(q, contexts)}")

        # 路由 3：闲聊 → 直接生成
        return self._generate(q)

    # ------------------------------------------------------------------
    def _generate(self, question: str, context: str | None = None, max_new: int = 50) -> str:
        """用 ch04 推理引擎生成（小模型尽力而为——见文档 8.4 的诚实讨论）。"""
        prompt = f"Background: {context}\n" if context else ""
        prompt += f"Q: {question}\nA: "
        prompt = sanitize(prompt, self.tokenizer)
        prompt = prompt[: self.cfg["max_len"] - max_new - 2]
        ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_new,
                                  temperature=0.8, top_k=40)
        text = self.tokenizer.decode(out[0].tolist())
        # 去掉 prompt 复读部分
        if "A: " in text:
            text = text.split("A: ", 1)[1]
        return text.strip()[:80] or "（模型没能生成有效回答）"

    # ------------------------------------------------------------------
    def chat(self):
        """交互式对话循环。"""
        print("🤖 迷你 AI 助手（第 8 章大作业）就绪。输入 /help 看命令，/quit 退出。")
        while True:
            try:
                q = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not q:
                continue
            ans = self.answer(q)
            if ans == "__QUIT__":
                print("再见！")
                break
            print(f"助手> {ans}")


def main() -> None:
    parser = argparse.ArgumentParser(description="第 8 章：大作业——迷你 AI 助手")
    parser.add_argument("--ckpt", default="checkpoints/ch03-demo.pt")
    parser.add_argument("--ask", type=str, default=None, help="单次提问")
    parser.add_argument("--show-tools", action="store_true", help="展示工具路由")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assistant = MiniAssistant(args.ckpt, device)

    if args.ask:
        print(f"你> {args.ask}")
        print(f"助手> {assistant.answer(args.ask)}")
    else:
        assistant.chat()


if __name__ == "__main__":
    main()
