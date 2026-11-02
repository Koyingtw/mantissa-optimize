"""Which bit changes cost energy? (review round 6, N1). Energy per operation of the 15 FP64 layouts at a locked clock
(analysis/mech_toggle.py on runs mechb1300, mechb1700, mechn02) against per-bit-position flip counts between neighbouring
operands along k (pk) and across rows of A / columns of B (px), from `gemm_mant_bench --stats-only` (results/mech_bitpos.txt;
the data generators are deterministic, so these are the statistics of the measured operands).
Each candidate model is a least-squares fit of the energy change (%, relative to native) per run; reported are the
in-sample R^2 and the leave-one-layout-out R^2 (each layout predicted from a fit on the other 14).
usage: mech_regress.py <run id> [...]  → results/mech_regress.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIELDS = {"sgnexp": range(52, 64), "mhi": range(26, 52), "mlo": range(0, 26)}


def bitpos():
    rows = {}
    for line in open(f"{RES}/mech_bitpos.txt"):
        if not line.startswith("BITPOS"):
            continue
        low = re.search(r"low=(\w+)", line).group(1)
        frac = int(re.search(r"frac=(\d+)", line).group(1))
        pk = np.array(re.search(r"pk=\[([^\]]*)\]", line).group(1).split(), float)
        px = np.array(re.search(r"px=\[([^\]]*)\]", line).group(1).split(), float)
        hw = float(re.search(r"hw=([\d.]+)", line).group(1))
        rows[(frac, low)] = (pk, px, hw)
    return rows


def features(pk, px, hw):
    f = dict(k=pk.sum(), x=px.sum(), hw=hw)
    for name, r in FIELDS.items():
        f[f"k_{name}"] = pk[list(r)].sum()
        f[f"x_{name}"] = px[list(r)].sum()
    # position-weighted: weight 2^(b-52) of the significand for mantissa bits, 1 for sign/exponent bits
    w = np.array([2.0 ** (b - 52) if b < 52 else 1.0 for b in range(64)])
    f["k_w"], f["x_w"] = (pk * w).sum(), (px * w).sum()
    return f


MODELS = {
    "total flips": ["tot"],
    "set bits": ["hw"],
    "k, row separately": ["k", "x"],
    "k, row, k x row": ["k", "x", "kx"],
    "fields x direction": ["k_sgnexp", "k_mhi", "k_mlo", "x_sgnexp", "x_mhi", "x_mlo"],
    "row fields only": ["x_sgnexp", "x_mhi", "x_mlo"],
    "row fields + k total": ["x_sgnexp", "x_mhi", "x_mlo", "k"],
    "position-weighted k, row": ["k_w", "x_w"],
}


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    bp = bitpos()
    m = pd.read_csv(f"{RES}/mech_toggle.csv")
    out = []
    for rid in sys.argv[1:]:
        g = m[m.id == rid].copy()
        F = pd.DataFrame([features(*bp[(r.frac, r.low)]) for r in g.itertuples()], index=g.index)
        g = pd.concat([g.drop(columns=[c for c in F.columns if c in g.columns]), F], axis=1)
        n0 = g[(g.frac == 52) & (g.low == "zero")].iloc[0]
        for c in F.columns:
            g[c] = g[c] - n0[c]  # changes relative to native
        g["tot"] = g.k + g.x
        g["kx"] = g.k * g.x
        y = g.de_pct.values
        for name, cols in MODELS.items():
            X = g[cols].values
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            loo = np.empty(len(y))
            for i in range(len(y)):
                keep = np.arange(len(y)) != i
                ci, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
                loo[i] = X[i] @ ci
            out.append(dict(id=rid, model=name, params=len(cols), r2_in=r2(y, X @ c), r2_loo=r2(y, loo),
                            max_loo_err=np.abs(loo - y).max(), coef=" ".join(f"{a}={v:.3g}" for a, v in zip(cols, c))))
    o = pd.DataFrame(out)
    o.to_csv(f"{RES}/mech_regress.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.groupby("model", sort=False)[["params", "r2_in", "r2_loo", "max_loo_err"]].mean().round(3).to_string())
    print(o[o.id == sys.argv[1]][["model", "coef"]].to_string(index=False))


if __name__ == "__main__":
    main()
