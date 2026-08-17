"""mcp.py — 第 7 章：MCP 风格工具协议（教学版，约 220 行）

MCP（Model Context Protocol）= 模型上下文协议，是 2025 年后 Agent 工具调用的
标准协议（Anthropic 发起、OpenAI 跟进）。核心思想：把「工具」标准化成
「可描述的、可校验的、可远程调用的接口」——模型不用知道工具怎么实现，
只需要知道工具的 schema（名字、描述、参数格式）。

本章教学实现一个极简 MCP：
  - ToolSchema：JSON Schema 风格的参数描述（name/type/required）
  - 工具注册与参数校验（缺参/类型错 → 结构化错误）
  - 类 JSON-RPC 的请求/响应协议（{jsonrpc, id, method, params}）
  - 安全护栏：文件工具路径白名单（只读项目内文件）

用法：
    .venv/Scripts/python.exe code/ch07/mcp.py --demo          # 完整请求/响应演示
    .venv/Scripts/python.exe code/ch07/mcp.py --server        # 启动「服务器」（模拟）
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import operator
import sys
from pathlib import Path


# ======================================================================
# 7.1 工具 Schema：给工具一张「身份证」
# ======================================================================
class ToolSchema:
    """描述一个工具的参数格式（JSON Schema 子集）。"""

    def __init__(self, name: str, description: str,
                 parameters: dict, required: list[str] | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters          # {参数名: {"type": "string|number|integer", "description": ...}}
        self.required = required or []

    def validate(self, args: dict) -> str | None:
        """校验参数：缺必填/类型错 → 返回错误信息；合法返回 None。"""
        for r in self.required:
            if r not in args:
                return f"缺少必填参数 {r!r}"
        for key, val in args.items():
            spec = self.parameters.get(key)
            if spec is None:
                continue                     # 多余参数忽略（真实系统可能拒绝）
            t = spec["type"]
            ok = {"string": isinstance(val, str),
                  "number": isinstance(val, (int, float)) and not isinstance(val, bool),
                  "integer": isinstance(val, int) and not isinstance(val, bool),
                  "boolean": isinstance(val, bool)}[t]
            if not ok:
                return f"参数 {key!r} 类型应为 {t}，实际 {type(val).__name__}"
        return None


# ======================================================================
# 7.2 工具实现（安全版本）
# ======================================================================
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_calc(expr: str) -> float:
    """复查#2：与 ch06 同款守卫——幂限指数、结果限幅，防 9**9**9 类 DoS。"""
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            left, right = _eval(n.left), _eval(n.right)
            if type(n.op) is ast.Pow:
                if not isinstance(right, (int, float)) or not -20 <= right <= 20:
                    raise ValueError("幂运算指数仅支持 -20~20（防溢出）")
            result = _OPS[type(n.op)](left, right)
            if abs(result) > 1e15:
                raise ValueError("结果超出安全范围（±1e15）")
            return result
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](_eval(n.operand))
        raise ValueError(f"不允许的表达式：{expr}")
    return _eval(node)


class McpServer:
    """极简 MCP 服务器：注册工具 → 校验参数 → 分发调用 → 返回结构化结果。"""

    # ch07/mcp.py -> code/ch07 -> code -> 项目根（三层 parent）
    ALLOWED_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self):
        self.tools: dict[str, tuple[ToolSchema, callable]] = {}
        self._register_core_tools()

    def register(self, schema: ToolSchema, fn: callable):
        self.tools[schema.name] = (schema, fn)

    def _register_core_tools(self):
        self.register(ToolSchema(
            "calc", "四则运算计算器",
            {"expression": {"type": "string", "description": "数学表达式，如 2+3*4"}},
            required=["expression"]), lambda expression: _safe_calc(expression))
        self.register(ToolSchema(
            "file_read", "读取项目内文本文件（只读，路径限项目内）",
            {"path": {"type": "string", "description": "相对项目根目录的路径，如 docs/chapter-01-tokenizer.md"}},
            required=["path"]), self._file_read)
        self.register(ToolSchema(
            "now", "当前日期时间", {}, []),
            lambda: _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---- 安全护栏：路径白名单 ----
    def _file_read(self, path: str) -> str:
        p = (self.ALLOWED_ROOT / path).resolve()
        if ":" in p.name:                                   # 审查 M2 修复：NTFS ADS 流绕过
            raise PermissionError(f"拒绝 ADS 路径：{path}")
        if not p.is_relative_to(self.ALLOWED_ROOT):       # 越界检查（防 ../ 逃逸）
            raise PermissionError(f"路径越界：{path} 不在项目目录内")
        if not p.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        return p.read_text(encoding="utf-8")[:2000]       # 长度上限，防撑爆上下文

    # ==================================================================
    # 7.3 类 JSON-RPC 协议：请求/响应
    # ==================================================================
    def handle(self, request: dict) -> dict:
        """处理一个协议请求，返回协议响应。"""
        if not isinstance(request, dict):       # 复查#3：request 非 dict 容错
            request = {}
        method = request.get("method")
        rid = request.get("id")
        params = request.get("params", {})
        if not isinstance(params, dict):        # 审查 M1 修复：params 非 dict 时容错
            params = {}

        if method == "list_tools":
            return {"jsonrpc": "2.0", "id": rid, "result": self.list_tools()}

        if method == "call_tool":
            name = params.get("name")
            args = params.get("arguments", {})
            if not isinstance(args, dict):      # 复查#3：arguments 非 dict 容错
                args = {}
            if name not in self.tools:
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32601, "message": f"未知工具：{name}"}}
            schema, fn = self.tools[name]
            try:                                # validate 与调用同进 try（原 validate 在 try 外可崩）
                err = schema.validate(args)
                if err:
                    return {"jsonrpc": "2.0", "id": rid,
                            "error": {"code": -32602, "message": err}}
                allowed = set(schema.parameters)              # P1-4：只传 schema 声明的参数（多余忽略）
                kwargs = {k: v for k, v in args.items() if k in allowed}
                result = fn(**kwargs)
                return {"jsonrpc": "2.0", "id": rid, "result": str(result)}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}}
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"未知方法：{method}"}}

    def list_tools(self) -> list[dict]:
        return [{"name": s.name, "description": s.description,
                 "parameters": s.parameters, "required": s.required}
                for s, _ in self.tools.values()]


# ======================================================================
# 7.4 客户端：发请求 + 解析响应
# ======================================================================
class McpClient:
    def __init__(self, server: McpServer):
        self.server = server
        self._next_id = 1

    def list_tools(self) -> list[dict]:
        return self.server.handle({"jsonrpc": "2.0", "id": self._next_id,
                                   "method": "list_tools"})["result"]

    def call(self, name: str, **args) -> str:
        resp = self.server.handle({"jsonrpc": "2.0", "id": self._next_id,
                                   "method": "call_tool",
                                   "params": {"name": name, "arguments": args}})
        self._next_id += 1
        if "error" in resp:
            raise RuntimeError(f"[{resp['error']['code']}] {resp['error']['message']}")
        return resp["result"]


# ======================================================================
def demo() -> None:
    server = McpServer()
    client = McpClient(server)
    print("===== list_tools（模型的工具菜单）=====")
    for t in client.list_tools():
        print(f"  · {t['name']} — {t['description']}")
        print(f"    参数: {t['parameters']} 必填: {t['required']}")

    print("\n===== call_tool 正常调用 =====")
    print("calc(2+3*4) =", client.call("calc", expression="2+3*4"))
    print("now() =", client.call("now"))
    print("file_read(docs/chapter-01-tokenizer.md) 前 30 字 =",
          client.call("file_read", path="docs/chapter-01-tokenizer.md")[:30])

    print("\n===== 错误处理（协议层）=====")
    for bad in [{"jsonrpc": "2.0", "id": 1, "method": "call_tool",
                 "params": {"name": "calc"}},                        # 缺参
                {"jsonrpc": "2.0", "id": 2, "method": "call_tool",
                 "params": {"name": "calc", "arguments": {"expression": 42}}},  # 类型错
                {"jsonrpc": "2.0", "id": 3, "method": "call_tool",
                 "params": {"name": "file_read", "arguments": {"path": "../../Windows/system32/drivers/etc/hosts"}}},  # 越界
                {"jsonrpc": "2.0", "id": 4, "method": "nope"}]:       # 未知方法
        print(f"  {bad['params'] if 'params' in bad else bad['method']} → "
              f"{json.dumps(server.handle(bad), ensure_ascii=False)[:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="第 7 章：MCP 风格工具协议")
    parser.add_argument("--demo", action="store_true", help="跑完整演示")
    parser.add_argument("--server", action="store_true", help="启动服务器（stdin 一行一个 JSON）")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.server:
        server = McpServer()
        print("MCP 服务器就绪：stdin 输入 JSON 请求（Ctrl+C 退出）", file=sys.stderr)
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip():
                try:
                    print(json.dumps(server.handle(json.loads(line)), ensure_ascii=False))
                except json.JSONDecodeError as e:
                    print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
