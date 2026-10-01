"""Exponent-spread experiment (e1_gemm_tdp.sbatch with SPREADS): does a wide exponent range hurt the
format-changing methods (FP64 emulation on INT8 tensor cores, FP32/TF32) while mantissa truncation, which
keeps FP64's exponent, is unaffected?

usage: range_sweep.py <jobid> [...]  -> results/range_sweep_raw.csv, results/range_sweep.csv, tables on stdout
       (RANGE_OUT=<prefix> overrides results/range_sweep)

Per (cap, spread, spread mode, method): TFLOP/s averaged over the ABBA passes, the Frobenius relative error
(dominated by the largest entries), the median and 99th percentile of the elementwise relative error (where the
small entries show up), the bits cuBLAS emulation chose (dynamic mode), and the time relative to truncation to
36 bits (T_method / T_trunc36, > 1 = truncation faster).
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi, window_stats  # noqa: E402

TRUNC = ("fp64", 36)
NUM = ("tdp", "frac", "m", "n", "k", "tflops", "rel_err", "emu_bits", "spread", "err_p50", "err_p99",
       "t_begin", "t_end")


def label(r):
    dt, fr = r["dtype"], int(r["frac"])   # r.dtype would be the Series' own dtype
    return {"fp64": f"截斷 FP64 {fr} 位" if fr < 52 else "原生 FP64", "fp64emu": f"cuBLAS 模擬 {fr} 位",
            "fp64emud": "cuBLAS 模擬（動態）", "gemmul8": f"GEMMul8 {fr} 模數", "fp32": "FP32",
            "tf32": "TF32"}.get(dt, dt)


def load(jobs, logdir="logs"):
    rows = []
    for j in jobs:
        smi_path = f"{logdir}/e1_{j}_smi.csv"
        smi = load_smi(smi_path) if os.path.exists(smi_path) else None
        for line in open(f"{logdir}/e1_{j}.out"):
            if not line.startswith("RESULT"):
                continue
            kv = dict(re.findall(r"(\w+)=([^\s,]+)", line.replace("tag=", "")))
            r = {"job": j, "host": kv["host"], "pass_": kv["pass"], "dtype": kv["dtype"], "smode": kv.get("smode", "elem")}
            for f in NUM:
                r[f] = float(kv[f]) if f in kv else float("nan")
            if smi is not None:
                w = window_stats(smi, r["t_begin"], r["t_end"], busy_gpu_w=150)
                r.update(mod_w=w["mod_w"], sm_mhz=w["sm_mhz"])
            rows.append(r)
    d = pd.DataFrame(rows)
    d["tdp"] = d.tdp.astype(int)
    d["frac"] = d.frac.astype(int)
    d.loc[d.spread == 0, "smode"] = "elem"   # no spread: the mode is irrelevant
    return d


def main(jobs):
    d = load(jobs)
    out = os.environ.get("RANGE_OUT", "results/range_sweep")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    d.to_csv(out + "_raw.csv", index=False)
    key = ["tdp", "spread", "smode", "dtype", "frac"]
    agg = {"tflops": "mean", "rel_err": "max", "err_p50": "max", "err_p99": "max", "emu_bits": "max"}
    if "mod_w" in d:
        agg.update(mod_w="mean", sm_mhz="mean")
    g = d.groupby(key).agg(agg).reset_index()
    g["method"] = g.apply(label, axis=1)
    tr = g[(g.dtype == TRUNC[0]) & (g.frac == TRUNC[1])].set_index(["tdp", "spread", "smode"]).tflops.rename("trunc")
    nat = g[(g.dtype == "fp64") & (g.frac == 52)].set_index(["tdp", "spread", "smode"]).tflops.rename("native")
    g = g.join(tr, on=["tdp", "spread", "smode"]).join(nat, on=["tdp", "spread", "smode"])
    g["t_vs_trunc36"] = g.trunc / g.tflops
    g["trunc36_gain_pct"] = (g.trunc / g.native - 1) * 100
    g.to_csv(out + ".csv", index=False)
    pd.set_option("display.width", 250)
    fmt = lambda v: f"{v:.2e}" if abs(v) < 1e-2 else f"{v:.3g}"  # noqa: E731
    for mode in sorted(g.smode.unique()):
        gm = g[g.smode == mode]
        print(f"\n== 跨度模式 {mode}：逐元素相對誤差中位數（列 = 方法，欄 = 跨度，2 的次方數）")
        print(gm.pivot_table(index="method", columns="spread", values="err_p50", aggfunc="max").to_string(float_format=fmt))
        print(f"\n== 跨度模式 {mode}：逐元素相對誤差 p99")
        print(gm.pivot_table(index="method", columns="spread", values="err_p99", aggfunc="max").to_string(float_format=fmt))
        for tdp, gt in gm.groupby("tdp"):
            print(f"\n== 跨度模式 {mode}，{tdp} W：時間 ÷ 截斷 36 位（> 1 表示截斷較快）")
            print(gt.pivot_table(index="method", columns="spread", values="t_vs_trunc36").to_string(float_format=lambda v: f"{v:.2f}"))
        if (gm.dtype == "fp64emud").any():
            print(f"\n== 跨度模式 {mode}：cuBLAS 動態模式選用的位數")
            print(gm[gm.dtype == "fp64emud"].pivot_table(index="tdp", columns="spread", values="emu_bits").to_string())
    print("\n== 截斷 36 位相對原生的加速（%）")
    print(g[(g.dtype == TRUNC[0]) & (g.frac == TRUNC[1])].pivot_table(index=["smode", "spread"], columns="tdp",
                                                                   values="trunc36_gain_pct").to_string(float_format=lambda v: f"{v:+.1f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
