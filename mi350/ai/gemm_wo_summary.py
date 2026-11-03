"""W1: BF16 / FP8 GEMM at LLM shapes with both operands or only the weights truncated (scripts/ai_gemm_wo.sh).

Pairs each window with the native window (BF16 7 bits, FP8 3 bits) of the same cap, pass, GPU and shape, as
analysis/e1_summary.py does, but keeps the shape in the key (e1_summary pairs one shape per run). Mode comes from the
run id: aiwo_both (both operands), aiwo_bf16w (BF16, --only-b = weights), aiwo_fp8w (FP8, --only-a = weights).
usage: gemm_wo_summary.py <run id> [...]  (logs in ../logs) → results/ai_gemm_wo.csv, prints speedup by cap × shape
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from e1_summary import ci, load  # noqa: E402

NATIVE = {"bf16": 7, "fp8": 3}
SHAPE = {(8192, 14336, 4096): "up", (8192, 4096, 14336): "down"}


def main():
    out = []
    for rid in sys.argv[1:]:
        mode = "both" if "_both_" in rid else "weights"
        d = load([rid], os.path.join(HERE, "..", "logs"))
        key = ["cap", "pass_", "gpu", "m", "n", "k", "dtype"]
        for dt, nb in NATIVE.items():
            g = d[d.dtype == dt]
            if g.empty:
                continue
            nat = g[g.frac == nb].set_index(key)
            for (cap, frac, m, n, k), h in g[g.frac < nb].groupby(["cap", "frac", "m", "n", "k"]):
                b = nat.reindex(h.set_index(key).index)
                r = h.tflops.values / b.tflops.values - 1
                lo, hi = ci(r)
                out.append(dict(mode=mode, dtype=dt, frac=frac, cap=cap, shape=SHAPE.get((m, n, k), f"{m}x{n}x{k}"),
                                n=int(np.isfinite(r).sum()), speedup_pct=100 * np.nanmean(r), lo=100 * lo, hi=100 * hi,
                                tflops_native=b.tflops.mean(), power_w=h.power_acc_w.mean(),
                                power_native_w=b.power_acc_w.mean(), gfxclk=h.gfxclk.mean(), gfxclk_native=b.gfxclk.mean(),
                                run=rid))
    s = pd.DataFrame(out).sort_values(["dtype", "cap", "shape", "frac", "mode"], ascending=[True, False, True, False, True])
    s.to_csv(os.path.join(HERE, "..", "results", "ai_gemm_wo.csv"), index=False)
    pd.set_option("display.width", 200)
    t = s.pivot_table(index=["dtype", "frac"], columns=["cap", "shape", "mode"], values="speedup_pct")
    print(t.sort_index(ascending=[True, False]).round(1).to_string())
    print("\nnative TFLOP/s:", s.groupby(["dtype", "cap", "shape"]).tflops_native.mean().round(0).to_dict())
    print("largest 95% half-width:", round(float(((s.hi - s.lo) / 2).max()), 2), "points")


if __name__ == "__main__":
    main()
