"""verify_ch03.py — 第 3 章预训练验证（用 run_all 的共享模型）。"""
from __future__ import annotations

import pathlib
import re
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
    py = sys.executable
    train = str(pathlib.Path(proj) / "code/ch03/train.py")

    # 训练（run_all 已训共享模型则跳过；独立运行才训练）
    if not pathlib.Path(ckpt).exists():
        r = subprocess.run([py, train, "--epochs", "400", "--d-model", "64",
                            "--n-layers", "2", "--block-size", "64", "--batch-size", "16",
                            "--ckpt", ckpt], capture_output=True, text=True, timeout=300)
        check("训练退出码 0", r.returncode == 0, f"(rc={r.returncode})")
        losses = [float(m) for m in re.findall(r"loss (\d+\.\d+)", r.stdout)]
        if len(losses) >= 2:
            check("loss 显著下降", losses[-1] < losses[0] * 0.5, f"{losses[0]:.3f} -> {losses[-1]:.3f}")
    else:
        print("  (共享模型已存在，跳过训练；直接验证 checkpoint/resume)")

    # checkpoint cfg + resume（resume 用临时副本，不污染共享模型）
    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch03"))
    import torch, shutil
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    check("ckpt 含 cfg", "cfg" in state and state["cfg"]["d_model"] == 64)
    tmp_ckpt = str(pathlib.Path(proj) / "tests/.cache/resume-test.pt")
    shutil.copy(ckpt, tmp_ckpt)
    r2 = subprocess.run([py, train, "--epochs", "50", "--d-model", "64",
                         "--n-layers", "2", "--block-size", "64", "--batch-size", "16",
                         "--ckpt", tmp_ckpt, "--resume"],
                        capture_output=True, text=True, timeout=300)
    check("--resume 恢复配置", r2.returncode == 0 and "已从 checkpoint 恢复配置" in r2.stdout)
    # P0-1 回归：ckpt 目录不存在时自动创建（复查#1 确认路径）
    deep_ckpt = str(pathlib.Path(proj) / "tests/.cache/no-such-dir/sub/ckpt.pt")
    r3 = subprocess.run([py, train, "--epochs", "50", "--d-model", "32",
                         "--n-layers", "1", "--block-size", "48", "--batch-size", "8",
                         "--ckpt", deep_ckpt],
                        capture_output=True, text=True, timeout=300)
    check("ckpt 目录自动创建", r3.returncode == 0 and pathlib.Path(deep_ckpt).exists())
    return passed, failed


if __name__ == "__main__":
    p, f = run(ckpt="tests/.cache/shared-model.pt")
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
