"""agent.py — 第 6 章：Agent 循环（ReAct）教学版，约 230 行

ReAct = Reasoning + Acting：模型交替输出「思考」和「行动」，
行动调用工具获得观察结果，循环直到给出最终答案。

```
Thought: 我需要算一下 2+3*4
Action: calc(2+3*4)
Observation: 14
Thought: 结果是 14，可以直接回答
Answer: 2+3*4 = 14
```

⚠️ 诚实说明：ReAct 的「大脑」（Thought 生成）在真实产品里是 API 大模型。
本章教学：
  - `--demo`：用内置「脚本大脑」驱动完整循环 → 看清机制（推荐先跑）
  - `--llm`：用第 3 章的小模型当大脑 → 看小模型的局限（生成乱码、解析失败、诚实终止）

用法：
    .venv/Scripts/python.exe code/ch06/agent.py --demo
    .venv/Scripts/python.exe code/ch06/agent.py --llm "2+3*4 等于多少"
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import operator
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch02"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch03"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch04"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch05"))
from transformer import MiniTransformer  # noqa: E402
from train import BUILTIN_CORPUS, CharTokenizer  # noqa: E402
from inference import load_model  # noqa: E402
from rag import KNOWLEDGE_BASE, CorpusStore  # noqa: E402


# ======================================================================
# 6.1 工具注册表：Agent 的「手脚」
# ======================================================================
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr: str):
    """只允许数字和四则运算的 eval（用 AST 白名单，拒绝任何函数/属性调用）。

    审查修复：幂运算限指数 0~20、结果限 ±1e15——防止 9**9**9 这类大整数幂 DoS。
    """
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = _eval(n.left), _eval(n.right)
            if type(n.op) is ast.Pow:
                if not isinstance(right, (int, float)) or not -20 <= right <= 20:
                    raise ValueError("幂运算指数仅支持 -20~20 的数值（防溢出）")
            result = _OPS[type(n.op)](left, right)
            if abs(result) > 1e15:
                raise ValueError("结果超出安全范围（±1e15）")
            return result
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_eval(n.operand))
        raise ValueError(f"不允许的表达式：{expr}")
    return _eval(node)


class ToolRegistry:
    """工具注册表：name -> (函数, 描述)。Agent 通过名字调用工具。"""

    def __init__(self):
        self.tools: dict[str, tuple] = {}
        self.register("calc", _safe_eval, "四则运算计算器，例：calc(2+3*4)")
        self.register("search", self._search, "知识库检索，例：search(什么是梯度下降)")
        self.register("now", lambda: _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      "当前日期时间，例：now()")
        self._store = CorpusStore(KNOWLEDGE_BASE)

    def register(self, name: str, fn, desc: str):
        self.tools[name] = (fn, desc)

    def _search(self, query: str) -> str:
        hits = self._store.retrieve(query, k=1)
        if not hits:
            return "未找到相关资料"
        title, content, score = hits[0]
        return f"[{title}](相似度{score:.2f}) {content}"

    def call(self, name: str, args_text: str) -> str:
        if name not in self.tools:
            return f"错误：未知工具 {name!r}，可用工具：{list(self.tools)}"
        fn, _ = self.tools[name]
        try:
            # 参数统一按字符串传，由工具自己解析
            if name == "calc":
                return str(fn(args_text.strip()))
            if name == "search":
                return str(fn(args_text.strip()))
            return str(fn())
        except Exception as e:
            return f"错误：{e}"


# ======================================================================
# 6.2 动作解析：从模型输出里提取 Action / Answer
# ======================================================================
# 审查 M5 修复 + 复查#5：逐行匹配（支持空参数 now()、多 Action 行、嵌套括号）
_ACTION_RE = re.compile(r"Action\s*:\s*(\w+)\s*\((.*)\)\s*$", re.I)
_ANSWER_RE = re.compile(r"Answer\s*:\s*(.+)", re.I)


def parse_action(text: str) -> tuple[str, str] | None:
    """解析 'Action: calc(2+3)' -> ('calc', '2+3')；找不到返回 None。

    复查#5：按行匹配——空参数 now()、多 Action 行、嵌套括号都支持；
    每行独立匹配，避免贪婪跨行吞并。
    """
    for line in text.splitlines():
        m = _ACTION_RE.search(line)
        if m:
            return m.group(1), m.group(2).strip()
    return None


def parse_answer(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


# ======================================================================
# 6.3 大脑抽象：脚本大脑（演示机制） vs LLM 大脑（小模型）
# ======================================================================
class ScriptedBrain:
    """脚本大脑：按预置的推理脚本行动，用于看清 ReAct 机制本身。"""

    def __init__(self):
        self.step = 0

    def act(self, observation: str | None, question: str) -> str:
        """根据观察结果产生下一步输出（Thought + Action/Answer）。"""
        self.step += 1
        if self.step == 1:
            return ("Thought: 我需要先算这个表达式。\n"
                    "Action: calc(2+3*4)")
        if self.step == 2:
            return ("Thought: 得到 14。再查一下相关知识丰富回答。\n"
                    "Action: search(梯度下降)")
        if self.step == 3:
            return ("Thought: 有了背景知识。现在可以给出最终答案。\n"
                    f"Answer: 2+3*4 = 14。另外，{observation}")

    def reset(self):
        self.step = 0


class LLMBrain:
    """LLM 大脑：用小模型生成下一步（教学展示：小模型能力有限）。"""

    def __init__(self, model: MiniTransformer, tokenizer: CharTokenizer,
                 device: str, max_new_tokens: int = 60):
        self.model, self.tokenizer = model, tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def act(self, observation: str | None, question: str) -> str:
        prompt = f"Q: {question}\n"
        if observation:
            prompt += f"Observation: {observation}\n"
        prompt += "Think step by step. If you need a tool, write 'Action: tool(args)'. "
        prompt += "When done, write 'Answer: ...'\nThought:"
        fallback = " " if " " in self.tokenizer.stoi else next(iter(self.tokenizer.stoi))
        prompt = "".join(c if c in self.tokenizer.stoi else fallback for c in prompt)
        ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        out = self.model.generate(ids, max_new_tokens=self.max_new_tokens,
                                  temperature=0.9, top_k=40)
        return self.tokenizer.decode(out[0].tolist())


# ======================================================================
# 6.4 Agent 主循环
# ======================================================================
class Agent:
    def __init__(self, tools: ToolRegistry, brain, max_steps: int = 5):
        self.tools = tools
        self.brain = brain
        self.max_steps = max_steps
        self.history: list[str] = []

    def run(self, question: str) -> str:
        print(f"\n🧠 问题：{question}")
        observation = None
        for step in range(1, self.max_steps + 1):
            text = self.brain.act(observation, question)          # 大脑输出
            self.history.append(text)
            print(f"\n── 第 {step} 步 ──\n{text}")

            action = parse_action(text)
            answer = parse_answer(text)
            if answer:                                             # 给出最终答案 → 结束
                print(f"\n✅ 最终答案：{answer}")
                return answer
            if action:
                name, args_text = action
                observation = self.tools.call(name, args_text)     # 执行工具
                print(f"🔧 工具 {name}({args_text}) → {observation}")
                continue
            # 大脑没说人话：重试（真实系统这里会引导模型重新输出）
            print("⚠️ 未解析到 Action/Answer，重试…")
            observation = "提示：请输出 Action: 工具名(参数) 或 Answer: 最终答案"

        print(f"\n⛔ 达到最大步数 {self.max_steps}，Agent 终止（诚实失败——大脑能力不足或任务超纲）")
        return None


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="第 6 章：Agent 循环（ReAct）")
    parser.add_argument("--demo", action="store_true", help="脚本大脑演示机制")
    parser.add_argument("--llm", type=str, default=None, metavar="问题", help="用小模型当大脑")
    parser.add_argument("--ckpt", default="checkpoints/ch03-demo.pt")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    tools = ToolRegistry()

    if args.demo:
        brain = ScriptedBrain()
    elif args.llm:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load_model(args.ckpt, device)
        tokenizer = CharTokenizer(BUILTIN_CORPUS)
        brain = LLMBrain(model, tokenizer, device)
        args.llm = args.llm.strip().strip('"')
    else:
        parser.print_help()
        return

    agent = Agent(tools, brain, max_steps=args.max_steps)
    agent.run(args.llm if args.llm else "2+3*4 等于多少？")

    print("\n本 Agent 的工具清单：")
    for name, (fn, desc) in tools.tools.items():
        print(f"  · {name} — {desc}")


if __name__ == "__main__":
    main()
