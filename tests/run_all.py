"""run_all.py — 全章节验证套件入口（canonical test command）

用法：
    .venv/Scripts/python.exe tests/run_all.py

流程：训练共享模型（ch03/05/08/09 需要）→ 逐个跑 verify_ch00~ch09 → 汇总。
每章模块也可独立运行：.venv/Scripts/python.exe tests/verify_ch02.py
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import time

PROJ = pathlib.Path(__file__).resolve().parent.parent
CACHE = pathlib.Path(__file__).resolve().parent / ".cache"
CKPT = CACHE / "shared-model.pt"
MODULES = [f"verify_ch{i:02d}" for i in range(10)]


def train_shared_model() -> None:
    if CKPT.exists():
        try:                            # 套件加固：损坏的共享模型重新训练
            import torch
            torch.load(CKPT, map_location="cpu", weights_only=True)
            return
        except Exception:
            print("共享模型损坏，重新训练…")
            CKPT.unlink(missing_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    print("训练共享测试模型（约 6 秒）…")
    r = subprocess.run(
        [sys.executable, str(PROJ / "code/ch03/train.py"),
         "--epochs", "400", "--d-model", "64", "--n-layers", "2",
         "--block-size", "64", "--batch-size", "16", "--ckpt", str(CKPT)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("❌ 共享模型训练失败：", r.stderr[-500:])
        sys.exit(1)


def main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    t0 = time.time()
    train_shared_model()

    grand = passed = failed = 0
    for name in MODULES:
        print(f"\n===== {name} =====")
        try:                            # 套件加固：单模块崩溃转 FAIL，不拖垮整个套件
            mod = importlib.import_module(name)
            p, f = mod.run(ckpt=str(CKPT), proj=str(PROJ))
        except Exception as e:
            print(f"  ❌ {name} 崩溃：{type(e).__name__}: {e}")
            p, f = 0, 1
        grand += p + f
        passed += p
        failed += f

    print(f"\n{'=' * 50}")
    print(f"总计：{passed} passed, {failed} failed（{grand} 项断言，"
          f"耗时 {time.time() - t0:.0f}s）")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
