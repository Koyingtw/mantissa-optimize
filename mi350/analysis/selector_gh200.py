"""Does the per-call selector carry over to GH200 (review round 6, N2)? Same rule and model form as selector.py, fitted
and cross-validated (leave one shape out) on the GH200 shape sweep (gh200/results/crossover_raw.csv: m = 8192,
n in 256..8192, k in 256..81920, module caps 1000 / 575 / 450 W; truncation to 36 bits, cuBLAS emulation with 32 bits,
GEMMul8 with 10 moduli). Tolerance classes from the paper: ~1e-11 allows truncation (36 bits) and cuBLAS emulation (32 bits);
~1e-9 also GEMMul8 with 10 moduli (1.0-8.2e-10).
usage: selector_gh200.py → results/selector_gh200.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GH = os.path.join(HERE, "..", "..", "gh200", "results")
RES = os.path.join(HERE, "..", "results")


def main():
    d = pd.read_csv(f"{GH}/crossover_raw.csv")
    d["method"] = np.select([(d.dtype == "fp64") & (d.frac == 52), (d.dtype == "fp64") & (d.frac == 36)],
                            ["native", "trunc36"], default=d.dtype + d.frac.astype(str))
    t = d.groupby(["tdp", "m", "n", "k", "method"]).tflops.mean().unstack("method").reset_index()
    emus = [c for c in t.columns if c not in ("tdp", "m", "n", "k", "native", "trunc36")]
    for c in ["trunc36"] + emus:
        t["r_" + c] = t["native"] / t[c]
    classes = {"1e-11": ["trunc36"] + [e for e in emus if "fp64emu" in e], "1e-9": ["trunc36"] + emus}
    X = lambda g: np.column_stack([np.ones(len(g)), 1 / g.k, 1 / g.m + 1 / g.n])  # noqa: E731
    rows = []
    shapes = t[["m", "n", "k"]].drop_duplicates().values.tolist()
    for m, n, k in shapes:
        test = t[(t.m == m) & (t.n == n) & (t.k == k)]
        train = t[~((t.m == m) & (t.n == n) & (t.k == k))]
        for r in test.itertuples():
            tr = train[train.tdp == r.tdp]
            pred = {"native": 1.0, "trunc36": tr.r_trunc36.mean()}
            for e in emus:
                c, *_ = np.linalg.lstsq(X(tr), tr["r_" + e], rcond=None)
                pred[e] = float((X(pd.DataFrame(dict(m=[m], n=[n], k=[k]))) @ c)[0])
            for cls, allowed in classes.items():
                opts = ["native"] + allowed
                meas = {o: (1.0 if o == "native" else getattr(r, "r_" + o)) for o in opts}
                pick = min(opts, key=lambda o: pred[o])
                best = min(opts, key=lambda o: meas[o])
                emu_opts = [o for o in allowed if o != "trunc36"]
                always_emu = min(emu_opts, key=lambda o: pred[o])
                rows.append(dict(tdp=r.tdp, m=m, n=n, k=k, tol=cls, pick=pick, best=best,
                                 loss=100 * (meas[pick] / meas[best] - 1), correct=pick == best,
                                 emu_loss=100 * (meas[always_emu] / meas[best] - 1), emu_correct=always_emu == best,
                                 trunc_loss=100 * (meas["trunc36"] / meas[best] - 1), trunc_correct=best == "trunc36"))
    o = pd.DataFrame(rows)
    o.to_csv(f"{RES}/selector_gh200.csv", index=False)
    print(o.groupby("tol").agg(cases=("correct", "size"), selector=("correct", "mean"), sel_loss=("loss", "mean"),
                               sel_max=("loss", "max"), always_emu=("emu_correct", "mean"), emu_loss=("emu_loss", "mean"),
                               always_trunc=("trunc_correct", "mean"), trunc_loss=("trunc_loss", "mean")).round(3).to_string())
    print(o.best.value_counts().to_string())


if __name__ == "__main__":
    main()
