"""verify_ch00.py — 第 0 章 micrograd 验证。独立可跑，也兼容 run_all.py。"""
from __future__ import annotations

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
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "code/ch00"))
    import micrograd

    # 梯度正确性：y = a*b + c
    a, b, c = micrograd.Value(3.0), micrograd.Value(4.0), micrograd.Value(2.0)
    y = a * b + c
    y.backward()
    check("dy/da == b", abs(a.grad - 4.0) < 1e-9)
    check("dy/db == a", abs(b.grad - 3.0) < 1e-9)
    check("dy/dc == 1", abs(c.grad - 1.0) < 1e-9)
    # 链式法则
    x = micrograd.Value(2.0)
    (x ** 2 * 3).backward()
    check("链式法则 dy/dx == 6x", abs(x.grad - 12.0) < 1e-9)
    # 训练收敛
    w = micrograd.Value(0.0)
    for _ in range(200):
        total = sum((w * xi + 2 - yt) ** 2 for xi, yt in [(1, 5), (2, 8), (3, 11), (4, 14)])
        total.backward()
        w.data -= 0.01 * w.grad
        w.grad = 0.0
    check("梯度下降收敛 w≈3", abs(w.data - 3.0) < 1e-6, f"(got {w.data:.6f})")
    return passed, failed


if __name__ == "__main__":
    p, f = run()
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
