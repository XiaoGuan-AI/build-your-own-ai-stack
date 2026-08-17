"""serve.py — 第 9 章：上线——极简 HTTP 推理服务（教学版，约 150 行）

把第 8 章的助手变成一个 HTTP 服务：任何程序（网页、App、脚本）都能调用。
零外部依赖——用 Python 标准库 http.server 实现。

端点：
    GET  /health             健康检查
    POST /generate           生成文本
         body: {"prompt": "2+3*4 等于多少", "max_new_tokens": 60,
                "temperature": 0.8, "top_k": 40}
         resp: {"text": "...", "tokens": 60, "time_ms": 123}

用法：
    .venv/Scripts/python.exe code/ch09/serve.py --port 18600
    curl -s -X POST http://127.0.0.1:18600/generate -H "Content-Type: application/json" \
         -d '{"prompt": "什么是梯度下降"}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch08"))
from assistant import MiniAssistant  # noqa: E402


class AssistantHandler(BaseHTTPRequestHandler):
    """HTTP 处理器：一个实例处理一个请求（ThreadingHTTPServer 并发）。"""

    assistant: MiniAssistant = None        # 类级共享（只加载一次模型）
    server_version = "MiniAssistant/1.0"

    # ---- 工具方法 ----
    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):      # 安静日志（少刷屏）
        sys.stderr.write(f"[serve] {time.strftime('%H:%M:%S')} {fmt % args}\n")

    # ---- 端点 ----
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": "ch03-demo"})
        elif self.path == "/generate":
            self._json(405, {"error": "method not allowed", "hint": "POST /generate"})
        else:
            self._json(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid JSON body"})
            return
        if "prompt" not in req or not isinstance(req["prompt"], str):
            self._json(400, {"error": "missing 'prompt' (string)"})
            return

        t0 = time.perf_counter()
        try:
            text = self.assistant.answer(req["prompt"])
        except ValueError as e:                     # 参数类错误 → 400 友好返回（不 500）
            self._json(400, {"error": str(e)})
            return
        dt_ms = (time.perf_counter() - t0) * 1000
        self._json(200, {"text": text, "time_ms": round(dt_ms, 1),
                         "tokens": len(text),       # P2：补齐 docstring 承诺的 tokens 字段
                         "prompt": req["prompt"][:40]})


def main() -> None:
    parser = argparse.ArgumentParser(description="第 9 章：极简 HTTP 推理服务")
    parser.add_argument("--ckpt", default="checkpoints/ch03-demo.pt")
    parser.add_argument("--port", type=int, default=18600, help="端口（5 位数）")
    args = parser.parse_args()

    # 模型只加载一次，所有请求共享（生产系统会用模型池/多副本）
    if not 1024 <= args.port <= 65535:
        print(f"❌ 端口必须在 1024~65535：{args.port}", file=sys.stderr)
        sys.exit(1)
    print("加载模型…")
    try:
        assistant = MiniAssistant(args.ckpt)
    except (FileNotFoundError, RuntimeError, AssertionError) as e:
        print(f"❌ 模型加载失败：{e}", file=sys.stderr)
        sys.exit(1)
    AssistantHandler.assistant = assistant
    print(f"✅ 模型就绪。服务启动：http://127.0.0.1:{args.port}")
    print("   GET  /health         健康检查")
    print("   POST /generate       生成文本（见文件头示例）")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), AssistantHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
