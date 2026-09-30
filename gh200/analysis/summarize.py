"""Summaries for the GH200 study: E1 (GEMM), E2 (HPL-MxP), E3 (HPL).

ΔW (等效功耗節省) is read off the native curve: for a reduced-mantissa point with
performance p, find the power the native configuration needs to reach p (linear
interpolation along the native curve, over both the TDP cap and the measured mean
module power), and subtract the reduced configuration's own value. When p is above
every native point the saving is only a lower bound and is marked with '>'.

usage: summarize.py --e1 1667,1668,1669 --hpl 1670 --mxp 1671 --out results/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_load import load_e1  # noqa: E402
from smi import load_smi, parse_hpl_log, window_stats  # noqa: E402


def native_power_for(perf, native, xcol):
    """Power (xcol) at which the native curve reaches `perf`; (value, is_lower_bound)."""
    n = native.sort_values("perf")
    if perf > n.perf.max():
        return n[xcol].iloc[-1], True
    if perf < n.perf.min():
        return np.nan, False
    return float(np.interp(perf, n.perf, n[xcol])), False


def add_delta_w(summary, key_cols, native_mask):
    out = []
    for key, grp in summary.groupby(key_cols):
        nat = grp[native_mask(grp)]
        for _, r in grp.iterrows():
            row = r.to_dict()
            for xcol, name in [("tdp", "dw_cap"), ("mod_w", "dw_meas")]:
                p, lb = native_power_for(r.perf, nat, xcol)
                row[name] = p - r[xcol]
                row[name + "_lb"] = lb
            out.append(row)
    return pd.DataFrame(out)


def summarize_e1(jobs, logdir):
    d = load_e1(jobs, logdir)
    s = (d.groupby(["dtype", "frac", "tdp"])
         .agg(perf=("tflops", "mean"), perf_sd=("tflops", "std"), n=("tflops", "size"),
              mod_w=("mod_w", "mean"), sm_mhz=("sm_mhz", "mean"), temp_max=("temp_max", "max"),
              thermal=("thermal_frac", "mean"), powercap=("powercap_frac", "mean"))
         .reset_index())
    s["gflops_per_w"] = s.perf * 1e3 / s.mod_w
    native = {"fp64": 52, "bf16": 7}
    base = s[s.frac == s.dtype.map(native)][["dtype", "tdp", "perf", "gflops_per_w"]]
    s = s.merge(base, on=["dtype", "tdp"], suffixes=("", "_native"))
    s["gain_pct"] = (s.perf / s.perf_native - 1) * 100
    s["eff_gain_pct"] = (s.gflops_per_w / s.gflops_per_w_native - 1) * 100
    s = add_delta_w(s, ["dtype"], lambda g: g.frac == g.dtype.map(native))
    return d, s


def summarize_hpl(job, logdir, name):
    log = f"{logdir}/{name}_{job}.out"
    r = parse_hpl_log(log)
    smi = load_smi(f"{logdir}/{name}_{job}_smi")
    stats = pd.DataFrame([window_stats(smi, x.t0, x.t1) for x in r.itertuples()])
    r = pd.concat([r, stats], axis=1)
    r["seconds"] = r.t1 - r.t0
    agg = dict(perf=("tflops", "mean"), perf_sd=("tflops", "std"), n=("tflops", "size"),
               residual=("residual", "max"), all_passed=("passed", "all"),
               mod_w=("mod_w", "mean"), sm_mhz=("sm_mhz", "mean"), energy_kj=("energy_kj", "mean"),
               thermal=("thermal_frac", "mean"), powercap=("powercap_frac", "mean"))
    if "lu_tflops" in r:
        agg.update(lu=("lu_tflops", "mean"), iters=("iters", "max"))
    s = r.groupby(["tdp", "variant"]).agg(**agg).reset_index()
    nat_variant = r.variant.iloc[0]  # the scripts list the native variant first
    base = s[s.variant == nat_variant][["tdp", "perf", "energy_kj"]]
    s = s.merge(base, on="tdp", suffixes=("", "_native"))
    s["gain_pct"] = (s.perf / s.perf_native - 1) * 100
    s["energy_saving_pct"] = (1 - s.energy_kj / s.energy_kj_native) * 100
    s["key"] = 0
    s = add_delta_w(s, ["key"], lambda g: g.variant == nat_variant).drop(columns="key")
    return r, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="")
    ap.add_argument("--hpl", default="")
    ap.add_argument("--mxp", default="")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if a.e1:
        raw, s = summarize_e1([int(j) for j in a.e1.split(",")], a.logdir)
        raw.to_csv(f"{a.out}/e1_raw.csv", index=False)
        s.to_csv(f"{a.out}/e1_summary.csv", index=False)
        print("== E1\n", s.round(3).to_string())
    for job, name, tag in [(a.hpl, "e3-hpl8", "e3"), (a.mxp, "e2-mxp8", "e2")]:
        if job:
            raw, s = summarize_hpl(int(job), a.logdir, name)
            raw.to_csv(f"{a.out}/{tag}_raw.csv", index=False)
            s.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
            print(f"== {tag}\n", s.round(3).to_string())


if __name__ == "__main__":
    main()
