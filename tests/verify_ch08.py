"""verify_ch08.py — 第 8 章大作业验证（路由三路径 + 命令）。"""
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
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch08"))
    import assistant

    a = assistant.MiniAssistant(ckpt, "cpu")
    check("数学路由 2+3*4=14", "14" in a.answer("2+3*4 等于多少"))
    check("数学路由连减", "5" in a.answer("10-2-3 是多少"))
    check("知识路由 RAG", "梯度下降" in a.answer("什么是梯度下降"))
    check("命令 /help", "/tools" in a.answer("/help"))
    check("命令 /tools", "calc" in a.answer("/tools"))
    check("命令 /quit", a.answer("/quit") == "__QUIT__")
    check("闲聊有回应", len(a.answer("你好呀")) > 0)

    py = sys.executable
    r = subprocess.run([py, str(pathlib.Path(proj) / "code/ch08/assistant.py"),
                        "--ckpt", ckpt, "--ask", "2+3*4 等于多少"],
                       capture_output=True, text=True, timeout=180)
    check("CLI 单问含答案", r.returncode == 0 and "14" in r.stdout)
    r2 = subprocess.run([sys.executable, str(pathlib.Path(proj) / "code/ch08/assistant.py"),
                         "--ckpt", ckpt, "--show-tools"],
                        capture_output=True, text=True, timeout=120)
    check("--show-tools 打印路由（P1-1）", r2.returncode == 0 and "路由" in r2.stdout)
    check("超窗 max_new 不崩（P0-2）", "（模型没能生成有效回答）" in a.answer("你好呀你好呀" * 30)
          or len(a.answer("你好呀你好呀" * 30)) > 0)
    return passed, failed


if __name__ == "__main__":
    p, f = run(ckpt="tests/.cache/shared-model.pt")
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
