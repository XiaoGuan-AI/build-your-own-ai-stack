"""verify_ch09.py — 第 9 章上线验证（HTTP 服务端到端）。"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
import urllib.request
import urllib.error

passed = failed = 0
PORT = 18996


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name} {detail}")
    else:
        failed += 1
        print(f"  FAIL  {name} {detail}")


def post(path, payload=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def run(ckpt: str | None = None, proj: str | None = None):
    global passed, failed
    passed = failed = 0
    proj = proj or str(pathlib.Path(__file__).resolve().parent.parent)
    py = sys.executable

    srv = subprocess.Popen([py, str(pathlib.Path(proj) / "code/ch09/serve.py"),
                            "--ckpt", ckpt, "--port", str(PORT)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
            body = json.loads(r.read().decode())
        check("/health 200 ok", r.status == 200 and body.get("status") == "ok")

        st, body = post("/generate", {"prompt": "2+3*4 等于多少"})
        check("/generate 200 含 14", st == 200 and "14" in body.get("text", ""),
              f"(status={st})")
        check("响应含 time_ms", isinstance(body.get("time_ms"), (int, float)))
        st, body = post("/generate", {"prompt": "什么是梯度下降"})
        check("知识问答", st == 200 and "梯度下降" in body.get("text", ""))
        st, body = post("/generate", {})
        check("缺 prompt → 400", st == 400)
        st, body = post("/generate", "not json")
        check("非法 JSON → 400", st == 400)
        ok = all("5" in post("/generate", {"prompt": "10-2-3 是多少"})[1].get("text", "")
                 for _ in range(3))
        check("3 连发稳定", ok)
    finally:
        srv.kill()
    return passed, failed


if __name__ == "__main__":
    p, f = run(ckpt="tests/.cache/shared-model.pt")
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
