"""Kernel-level calibration of the power model with an application's own operands (review round 7, P2).

scripts/kcal.sh runs, at a 1300 MHz lock and 1000 W, the GEMM microbenchmark on operands taken from the applications
(DF-MP2: the call i = 150 of the Taxol tensor; GP: trailing-update panels L21 at two steps and two ranges) in native
FP64 and with the applications' bit counts (round to nearest). From each window paired with the native window of the
same pass and GPU: R = (P_v - P0)/(P_n - P0), sigma = clk_v/clk_n, P0 = idle 262 W. Only the GEMM is truncated, so
for the program R_app = 1 - f (1 - R), sigma_app = 1 + f (sigma - 1), with f the GEMM's share of the native run time at
the same lock (scripts/gemmshare.sh: GP from HIP events around each trailing update, DF-MP2 from its per-call times).
These replace the whole-program locked-clock run of analysis/model_apps.py; the speedups predicted from them at 500-1000 W are compared with the
measured cap sweeps on node n02, next to the whole-program calibration (results/model_apps.csv) and to the fit on
synthetic-operand bit flips (results/switching_model_mp2.csv for DF-MP2).
usage: kernel_cal.py  (all kcal* runs in ../logs) → results/kernel_cal_R.csv, results/kernel_cal.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import load  # noqa: E402
from model_apps import P0, latest, native_perf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs")
APPS = {"DF-MP2": ("kcalmp2_", "mp2n02_", {32: "t32n", 20: "t20n"}),
        "GP": ("kcalgp_", "gpn02_", {36: "t36n", 28: "t28n"})}


def gemm_share():
    """f per app from the latest gemmshare log: second native pass of each GPU, mean over GPUs."""
    rid = sorted(f for f in os.listdir(LOG) if f.startswith("gemmshare_") and f.endswith(".out"))[-1]
    gp, mp2_ms, mp2_wall = {}, {}, {}
    for line in open(f"{LOG}/{rid}"):
        if line.startswith("PROF_GP"):
            # each GPU's process prints pass A, then pass B: keep the last
            gp[int(re.search(r"gpu=(\d+)", line).group(1))] = float(re.search(r"share=([\d.]+)", line).group(1))
        elif line.startswith("CALL") and "pass=B" in line:
            g = int(re.search(r"gpu=(\d+)", line).group(1))
            mp2_ms[g] = mp2_ms.get(g, 0) + float(re.search(r"ms=([\d.]+)", line).group(1))
        elif line.startswith("RESULT_MP2") and "pass=B" in line:
            g = int(re.search(r"gpu=(\d+)", line).group(1))
            mp2_wall[g] = float(re.search(r"wall=([\d.]+)", line).group(1))
    return {"GP": float(np.mean(list(gp.values()))), "DF-MP2": float(np.mean([mp2_ms[g] * 1e-3 / mp2_wall[g] for g in mp2_wall]))}


def main():
    share = gemm_share()
    # review round 9: the GEMM's share of the native *dynamic energy*, measured at phase boundaries in the program
    # (analysis/gp_energy.py), replaces the time share where available (GP: 0.97 against 0.90 in time)
    for app, f in (("GP", "gp_energy.csv"), ("DF-MP2", "mp2_energy.csv")):  # review round 10: both from energy
        if os.path.exists(f"{RES}/{f}"):
            share[app] = float(pd.read_csv(f"{RES}/{f}", index_col=0).loc["native", "w_E"])
    print("GEMM share of native run time:", {k: f"{v:.3f}" for k, v in share.items()})
    rows = []
    for app, (kp, _, bits) in APPS.items():
        for rid in sorted(f[:-4] for f in os.listdir(LOG) if f.startswith(kp) and f.endswith(".out")):
            d = load([rid], LOG)
            d = d[d.dtype == "fp64"]
            key = ["pass_", "gpu"]
            nat = d[d.frac == 52].set_index(key)
            for b in bits:
                g = d[d.frac == b].set_index(key)
                n = nat.reindex(g.index)
                R = (g.power_acc_w - P0) / (n.power_acc_w - P0)
                s = g.gfxclk / n.gfxclk
                rows.append(dict(app=app, run=rid, bits=b, variant=bits[b], n=len(g), R=R.mean(), R_sd=R.std(),
                                 sigma=s.mean(), P_native=n.power_acc_w.mean(), P_var=g.power_acc_w.mean(),
                                 tflops_native=n.tflops.mean(), dperf_pct=100 * (g.tflops / n.tflops - 1).mean()))
    r = pd.DataFrame(rows)
    r.to_csv(f"{RES}/kernel_cal_R.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # mean over operand samples per app and bit count; predictions at each cap
    whole = pd.read_csv(f"{RES}/model_apps.csv")
    out = []
    for (app, v), g in r.groupby(["app", "variant"]):
        R, sigma = g.R.mean(), g.sigma.mean()
        f = share[app]
        Ra, sa = 1 - f * (1 - R), 1 + f * (sigma - 1)
        sweep, _ = latest(APPS[app][1])
        curve = sweep[sweep.variant == "native"].assign(perf=lambda d: 1.0 / d.wall)[["cap", "perf"]]
        w = whole[(whole.app == app) & (whole.variant == v)].set_index("cap")
        for cap in sorted(sweep.cap.unique()):
            pn = native_perf(curve, cap)
            pred_raw = 100 * (sigma * native_perf(curve, P0 + (cap - P0) / R) / pn - 1)
            pred = 100 * (sa * native_perf(curve, P0 + (cap - P0) / Ra) / pn - 1)
            meas = float(sweep[(sweep.cap == cap) & (sweep.variant == v)].speedup_pct.iloc[0])
            out.append(dict(app=app, variant=v, cap=cap, share=f, R_kernel=R, R_kernel_min=g.R.min(),
                            R_kernel_max=g.R.max(), sigma_kernel=sigma, R_app=Ra, sigma_app=sa,
                            R_whole=w.R.iloc[0], sigma_whole=w.sigma.iloc[0], meas_pct=meas,
                            pred_raw=pred_raw, err_raw=pred_raw - meas,
                            pred_kernel=pred, err_kernel=pred - meas, pred_whole=w.loc[cap, "pred_pct"],
                            err_whole=w.loc[cap, "err_pts"]))
    p = pd.DataFrame(out)
    p.to_csv(f"{RES}/kernel_cal.csv", index=False)
    print(p.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for app, g in p.groupby("app"):
        print(f"{app}: kernel-level mean |err| {g.err_kernel.abs().mean():.2f} (max {g.err_kernel.abs().max():.2f}), "
              f"without the share {g.err_raw.abs().mean():.2f} (max {g.err_raw.abs().max():.2f}); "
              f"whole-program {g.err_whole.abs().mean():.2f} (max {g.err_whole.abs().max():.2f}) over {len(g)} points")
    print(f"all: kernel-level {p.err_kernel.abs().mean():.2f} (max {p.err_kernel.abs().max():.2f}); "
          f"whole-program {p.err_whole.abs().mean():.2f} (max {p.err_whole.abs().max():.2f})")


if __name__ == "__main__":
    main()
