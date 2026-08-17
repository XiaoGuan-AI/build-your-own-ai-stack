"""verify_ch06.py — 第 6 章 Agent 验证（解析/安全/循环）。"""
from __future__ import annotations

import pathlib
import subprocess
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
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch06"))
    import agent

    check("parse_action", agent.parse_action("Thought: x\nAction: calc(2+3*4)") == ("calc", "2+3*4"))
    check("parse_action 中文参数",
          agent.parse_action("Action: search(什么是梯度下降)") == ("search", "什么是梯度下降"))
    check("parse_action 无匹配返回 None", agent.parse_action("随便说") is None)
    check("parse_answer", agent.parse_answer("Thought: 好了\nAnswer: 结果是 14") == "结果是 14")

    tools = agent.ToolRegistry()
    check("calc 2+3*4=14", tools.call("calc", "2+3*4") == "14")
    check("calc 拒绝危险表达式", "错误" in tools.call("calc", "__import__('os').system('dir')"))
    check("calc 拒绝属性访问", "错误" in tools.call("calc", "(1).__class__"))
    check("search 命中知识库", "梯度下降" in tools.call("search", "什么是梯度下降"))
    check("未知工具报错", "未知工具" in tools.call("nope", ""))

    r = subprocess.run([str(pathlib.Path(proj) / ".venv/Scripts/python.exe"),
                        str(pathlib.Path(proj) / "code/ch06/agent.py"), "--demo"],
                       capture_output=True, text=True, timeout=120)
    check("ReAct demo 完整", r.returncode == 0 and "✅ 最终答案" in r.stdout)

    # 最大步数诚实失败
    brain = agent.ScriptedBrain()
    result = agent.Agent(agent.ToolRegistry(), brain, max_steps=1).run("测试")
    check("步数用尽返回 None", result is None)
    return passed, failed


if __name__ == "__main__":
    p, f = run()
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
