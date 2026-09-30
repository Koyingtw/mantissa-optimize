"""Truncation vs FP64 emulation crossover over GEMM shapes (e1_gemm_tdp.sbatch with SHAPES).

usage: crossover.py <jobid> [...]  -> results/crossover_raw.csv, results/crossover_fit.csv, table on stdout

Model: emulation pays, on top of its integer GEMM work (proportional to mnk), a cost to split and scale the
inputs (proportional to mk + nk) and a cost to combine the slice products into C (proportional to mn times
the number of slice pairs); truncated FP64 pays only the GEMM.  Dividing by the truncated GEMM time mnk/R:

    T_emu / T_trunc = A + B * (1/m + 1/n) + C * (1/k)

A = ratio of the per-flop GEMM rates, B = input splitting, C = output combining.  A, B, C are fitted per
power cap and method by least squares; emulation is faster where the ratio is below 1.  The single-feature
form A + B'/h with h = 1/(1/m + 1/n + 1/k) is also reported: it cannot tell input from output cost, and the
first real-shape points (a square and a long-k GEMM) already contradict it.
Speeds are TFLOP/s of the useful 2mnk flops, averaged over the ABBA passes on the same node.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi, window_stats  # noqa: E402

TRUNC = ("fp64", 36)
EMUS = {("fp64emu", 32): "cuBLAS 模擬 32 位", ("gemmul8", 10): "GEMMul8 10 模數"}


def load(jobs, logdir="logs"):
    rows = []
    for j in jobs:
        smi_path = f"{logdir}/e1_{j}_smi.csv"
        smi = load_smi(smi_path) if os.path.exists(smi_path) else None
        for line in open(f"{logdir}/e1_{j}.out"):
            if not line.startswith("RESULT"):
                continue
            kv = dict(re.findall(r"(\w+)=([^\s,]+)", line.replace("tag=", "")))
            r = dict(job=j, host=kv["host"], tdp=int(kv["tdp"]), pass_=kv["pass"], dtype=kv["dtype"],
                     frac=int(kv["frac"]), m=int(kv["m"]), n=int(kv["n"]), k=int(kv["k"]),
                     tflops=float(kv["tflops"]))
            if smi is not None:
                w = window_stats(smi, float(kv["t_begin"]), float(kv["t_end"]), busy_gpu_w=150)
                r.update(mod_w=w["mod_w"], sm_mhz=w["sm_mhz"])
            rows.append(r)
    return pd.DataFrame(rows)


def main(jobs):
    d = load(jobs)
    os.makedirs("results", exist_ok=True)
    d.to_csv("results/crossover_raw.csv", index=False)
    g = d.groupby(["tdp", "m", "n", "k", "dtype", "frac"]).tflops.mean().reset_index()
    key = ["tdp", "m", "n", "k"]
    nat = g[(g.dtype == "fp64") & (g.frac == 52)].set_index(key).tflops.rename("native")
    tr = g[(g.dtype == TRUNC[0]) & (g.frac == TRUNC[1])].set_index(key).tflops.rename("trunc")
    t = pd.concat([nat, tr], axis=1)
    fits = []
    for (dt, fr), label in EMUS.items():
        e = g[(g.dtype == dt) & (g.frac == fr)].set_index(key).tflops.rename("emu")
        x = t.join(e, how="inner").reset_index()
        x["method"] = label
        x["h"] = 1.0 / (1.0 / x.m + 1.0 / x.n + 1.0 / x.k)
        x["ratio"] = x.trunc / x.emu  # = T_emu / T_trunc (same useful flops)
        for tdp, xx in x.groupby("tdp"):
            if len(xx) < 4:  # three parameters: need more shapes than that
                continue
            def r2_of(pred):
                return 1 - ((xx.ratio - pred) ** 2).sum() / ((xx.ratio - xx.ratio.mean()) ** 2).sum()
            X = np.column_stack([np.ones(len(xx)), 1.0 / xx.m + 1.0 / xx.n, 1.0 / xx.k])
            (A, B, C), *_ = np.linalg.lstsq(X, xx.ratio.values, rcond=None)
            A1, B1 = np.polyfit(1.0 / xx.h, xx.ratio, 1)[::-1]  # single-feature form, for comparison
            fits.append(dict(method=label, tdp=tdp, A=A, B_in=B, C_out=C, r2=r2_of(X @ np.array([A, B, C])),
                             r2_h_only=r2_of(A1 + B1 / xx.h), points=len(xx)))
        x.to_csv(f"results/crossover_{dt}{fr}.csv", index=False)
        print(f"\n== {label}：T_模擬 / T_截斷36（< 1 表示模擬較快）")
        print(x.pivot_table(index=["n", "k"], columns="tdp", values="ratio").to_string(float_format=lambda v: f"{v:.2f}"))
    f = pd.DataFrame(fits)
    f.to_csv("results/crossover_fit.csv", index=False)
    print("\n== 擬合 ratio = A + B·(1/m+1/n) + C·(1/k)（r2_h_only：只用 h 的單一特徵版本）")
    print(f.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n== 截斷 36 位相對原生")
    print((t.trunc / t.native - 1).mul(100).unstack("tdp").to_string(float_format=lambda v: f"{v:+.1f}%"))


if __name__ == "__main__":
    main(sys.argv[1:])
