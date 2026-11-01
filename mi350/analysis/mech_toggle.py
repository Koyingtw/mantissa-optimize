"""Temporal-toggling controls on MI350X at a locked clock (review round 5, M2).

FP64 GEMM ($32768^2 \\times 2048$) with the dropped bits zeroed, set to ones, to one fixed pattern or to a random
pattern repeated along k (krep), and with the same multiset of values per row/column placed in sorted or shuffled order
along k. The bench prints, for every operand, the mean number of bits that change between consecutive k (tog) and the
mean number of set bits (hw). At a locked clock nothing is power-limited, so the energy per operation compares the
switching directly. Every window is paired with the native window (52 bits, zero layout) of the same pass and GPU.

usage: mech_toggle.py <run id> [...]   (logs in ../logs)  → results/mech_toggle.csv
Windows overlapping an unrelated job on GPU 0001:79 (my 2-s test run, Unix 1793527190-1793527215) are dropped.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import BDF, ci, load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs")
EXCLUDE = [(BDF.index("0000:79:00.0"), 1793527190, 1793527215)]


def main():
    ids = sys.argv[1:]
    if not ids:
        sys.exit(__doc__)
    d = load(ids, LOG)
    for g, a, b in EXCLUDE:
        d = d[~((d.gpu == g) & (d.t_end > a) & (d.t_begin < b))]
    d = d[d.dtype == "fp64"].copy()
    d["e_op"] = d.power_acc_w / d.tflops  # J per TFLOP
    key = ["id", "pass_", "gpu"]
    nat = d[(d.frac == 52) & (d.low == "zero")].set_index(key)
    out = []
    for (i, frac, low), g in d.groupby(["id", "frac", "low"]):
        b = nat.reindex(g.set_index(key).index)
        r = g.e_op.values / b.e_op.values - 1
        lo, hi = ci(r)
        out.append(dict(id=i, frac=frac, low=low, n=len(g), tog=g.tog.mean(), tox=g.tox.mean() if "tox" in g else np.nan,
                        hw=g.hw.mean(),
                        gfxclk=g.gfxclk.mean(), ppt_frac=g.ppt_frac.mean(), e_op=g.e_op.mean(),
                        de_pct=100 * np.nanmean(r), de_lo=100 * lo, de_hi=100 * hi,
                        dperf_pct=100 * np.nanmean(g.tflops.values / b.tflops.values - 1)))
    s = pd.DataFrame(out).sort_values(["id", "frac", "low"], ascending=[True, False, True])
    # least-squares fit of the energy change on the toggle and set-bit changes relative to native (per run id)
    fits = []
    for i, g in s.groupby("id"):
        n0 = g[(g.frac == 52) & (g.low == "zero")].iloc[0]
        tot = g.de_pct - g.de_pct.mean()

        def r2(cols):
            X = np.column_stack(cols)
            c, *_ = np.linalg.lstsq(X, g.de_pct, rcond=None)
            return 1 - ((g.de_pct - X @ c) ** 2).sum() / (tot ** 2).sum(), c
        row = dict(id=i)
        for name, cols in [("k", [g.tog - n0.tog]), ("hw", [g.hw - n0.hw]), ("k+hw", [g.tog - n0.tog, g.hw - n0.hw])]:
            row[f"r2_{name}"] = r2(cols)[0]
        if g.tox.notna().all() and (g.tox > 0).all():
            both = (g.tog + g.tox) - (n0.tog + n0.tox)
            row["r2_x"] = r2([g.tox - n0.tox])[0]
            row["r2_k+x"], c = r2([both])
            row["pct_per_flip"] = c[0]
            row["r2_k+x+hw"] = r2([both, g.hw - n0.hw])[0]
        fits.append(row)
    s.to_csv(f"{RES}/mech_toggle.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(pd.DataFrame(fits).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
