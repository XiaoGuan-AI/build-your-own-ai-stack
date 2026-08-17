"""verify_ch07.py — 第 7 章 MCP 验证（schema/协议/护栏）。"""
from __future__ import annotations

import json
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
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch07"))
    import mcp

    server = mcp.McpServer()
    client = mcp.McpClient(server)
    names = {t["name"] for t in client.list_tools()}
    check("三个核心工具", {"calc", "file_read", "now"} <= names, f"(got {sorted(names)})")
    check("calc 调用", client.call("calc", expression="2+3*4") == "14")
    check("file_read 白名单内", "# 🛠️" in client.call("file_read", path="README.md"))
    check("now 返回日期", client.call("now").startswith("20"))

    def handle(req):
        return server.handle(req)
    r = handle({"jsonrpc": "2.0", "id": 1, "method": "call_tool", "params": {"name": "calc"}})
    check("缺参 → -32602", r["error"]["code"] == -32602)
    r = handle({"jsonrpc": "2.0", "id": 2, "method": "call_tool",
                "params": {"name": "calc", "arguments": {"expression": 42}}})
    check("类型错 → -32602", r["error"]["code"] == -32602)
    r = handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})
    check("未知方法 → -32601", r["error"]["code"] == -32601)
    r = handle({"jsonrpc": "2.0", "id": 4, "method": "call_tool",
                "params": {"name": "file_read", "arguments": {"path": "../README.md"}}})
    check("路径越界被拒", "error" in r and "越界" in r["error"]["message"])
    r = handle({"jsonrpc": "2.0", "id": 5, "method": "call_tool",
                "params": {"name": "calc", "arguments": {"expression": "__import__('os').system('x')"}}})
    check("危险表达式被拒", "error" in r)
    r = handle({"jsonrpc": "2.0", "id": 9, "method": "call_tool", "params": 42})
    check("params 非 dict 不崩", "error" in r or "result" in r,
          f"(got {json.dumps(r, ensure_ascii=False)[:60]})")
    r = handle([1, 2])                        # 复查#3：request 非 dict
    check("request 非 dict 不崩", "error" in r or "result" in r)
    r = handle({"jsonrpc": "2.0", "id": 11, "method": "call_tool",
                "params": {"name": "calc", "arguments": 42}})
    check("arguments 非 dict 不崩", "error" in r or "result" in r,
          f"(got {json.dumps(r, ensure_ascii=False)[:60]})")
    r = handle({"jsonrpc": "2.0", "id": 12, "method": "call_tool",
                "params": {"name": "calc", "arguments": {"expression": "9**9**9"}}})
    check("MCP calc DoS 被拒（复查#2）", "error" in r, f"(got {r.get('error', {}).get('message', '')[:40]})")
    r = handle({"jsonrpc": "2.0", "id": 13, "method": "call_tool",
                "params": {"name": "calc", "arguments": {"expression": "1+1", "extra": 123}}})
    check("多余参数忽略（P1-4）", r.get("result") == "2", f"(got {r.get('result')!r})")
    r = handle({"jsonrpc": "2.0", "id": 10, "method": "call_tool",
                "params": {"name": "file_read", "arguments": {"path": "README.md:stream"}}})
    check("NTFS ADS 流被拒", "error" in r and "ADS" in r["error"]["message"],
          f"(got {r.get('error', {}).get('message', '')[:40]})")

    # server 模式（stdout 纯协议）
    py = sys.executable
    p = subprocess.Popen([py, str(pathlib.Path(proj) / "code/ch07/mcp.py"), "--server"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    out, _ = p.communicate(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "list_tools"}) + "\n",
                           timeout=30)
    resp = json.loads(out)
    check("server 模式响应合法", resp["id"] == 1 and len(resp["result"]) == 3)
    p.kill()
    return passed, failed


if __name__ == "__main__":
    p, f = run()
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
