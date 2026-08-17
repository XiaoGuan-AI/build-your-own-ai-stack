"""verify_ch04.py — 第 4 章推理引擎验证（采样策略 + 端到端）。"""
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

    sys.path.insert(0, str(pathlib.Path(proj) / "code/ch04"))
    import torch
    import torch.nn.functional as F
    import inference

    # 采样策略单元（固定 logits）
    lg = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    check("top_k=3 只采前 3", inference.sample_token(lg.clone(), top_k=3).item() <= 2)
    check("top_p=0.85 只采前 2", inference.sample_token(lg.clone(), top_p=0.85).item() <= 1)
    p_low = F.softmax(lg / 0.5, dim=-1)
    p_high = F.softmax(lg / 2.0, dim=-1)
    check("低温分布更尖", p_low[0, 0] > p_high[0, 0])

    # 端到端 CLI（用共享模型）
    r = subprocess.run([py, str(pathlib.Path(proj) / "code/ch04/inference.py"),
                        "--ckpt", ckpt, "--prompt", "锄禾日当午",
                        "--max-new-tokens", "30", "--temperature", "0.8", "--top-k", "40"],
                       capture_output=True, text=True, timeout=180)
    # OOV prompt 端到端（专家2：原 CLI 测试全在词表内，M4 兜底无断言保护）
    r_ov = subprocess.run([py, str(pathlib.Path(proj) / "code/ch04/inference.py"),
                           "--ckpt", ckpt, "--prompt", "量子纠缠与虫洞？？",
                           "--max-new-tokens", "20", "--temperature", "0.8", "--top-k", "40"],
                          capture_output=True, text=True, timeout=180)
    check("OOV prompt 不崩（M4 兜底）", r_ov.returncode == 0, f"(rc={r_ov.returncode})")
    check("CLI 退出码 0", r.returncode == 0, f"(rc={r.returncode})")
    m = re.search(r"【生成】(.*)", r.stdout)
    check("生成非空", m is not None and len(m.group(1)) > 5)

    # 审查 M8 回归：temperature=0 明确报错（原产生 NaN 采样）
    try:
        inference.sample_token(torch.tensor([[1.0, 2.0]]), temperature=0)
        check("temperature=0 报错", False)
    except ValueError:
        check("temperature=0 报错", True)
    return passed, failed


if __name__ == "__main__":
    p, f = run(ckpt="tests/.cache/shared-model.pt")
    print(f"===== {p} passed, {f} failed =====")
    sys.exit(1 if f else 0)
