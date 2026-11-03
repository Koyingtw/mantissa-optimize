"""From an application tolerance to a GEMM setting and back, on the Gaussian-process likelihood (review round 7, P5).

1. Calibrate: the largest log-likelihood deviation |dl| of earlier truncated runs (gpemu64k: n = 64000, nb = 1024,
   28 and 36 kept bits, all caps and GPUs) divided by the measured GEMM product error of that bit count (iso-accuracy
   sweep, largest over k) gives kappa; we keep the larger of the two (conservative).
2. Map: an application tolerance tau on |dl| becomes a product tolerance eps = tau / kappa; each method takes its
   cheapest measured setting with error <= eps (kept bits from 20..44 in steps of 4, moduli from 8..14), as the
   selector does, and the selector predicts which is faster from the GEMM sweep at k = 2048 (pareto2048) per cap.
3. Verify: a separate run (gptol2048: n = 64000, nb = 2048, both methods at every setting, 1000 and 500 W) gives the
   measured |dl| of each setting, to be <= tau, and the measured faster method.
Review round 8: a kappa from truncated runs overstates emulation's likelihood error about tenfold, so the selector picks
more moduli than needed. Scheme "per-method" calibrates emulation's own kappa_e from earlier emulated runs (gpsel, 12
moduli, block sizes other than 2048, so independent of the verification run) and verifies the moduli it then picks
(gptolemu2048: 9 and 12 moduli; 11 is in gptol2048).
usage: gptol.py → results/gptol.csv
"""
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs")
TAUS = (1e-2, 1e-4, 1e-6)


def runs(rid):
    rows = []
    for line in open(f"{LOG}/{rid}.out"):
        if line.startswith("RESULT_GP"):
            kv = dict(re.findall(r"(\w+)=([^\s,\[]+)", line.replace("tag=", "")))
            dll = [abs(float(x)) for x in re.search(r"dll=\[([^\]]*)\]", line).group(1).split()]
            rows.append(dict(job=rid, cap=int(kv["cap"]), round=int(kv["round"]), gpu=int(kv["gpu"]), variant=kv["variant"],
                             ok=kv["status"] == "ok", wall=float(kv["wall"].rstrip("s")), dll=max(dll)))
    return pd.DataFrame(rows)


def latest(prefix):
    return sorted(f[:-4] for f in os.listdir(LOG) if f.startswith(prefix) and f.endswith(".out"))[-1]


def main():
    par = pd.concat([pd.read_csv(f"{RES}/{f}") for f in os.listdir(RES)
                     if f.startswith("pareto") and f.endswith("_summary.csv")])
    err = par.groupby(["dtype", "frac"]).rel_err.max()
    cal = runs(latest("gpemu64k_"))
    kap = {b: cal[cal.variant == f"t{b}n"].dll.max() / err[("fp64", b)] for b in (28, 36)}
    kappa = max(kap.values())
    print("kappa per bit count", {b: f"{v:.3g}" for b, v in kap.items()}, "-> kappa", f"{kappa:.3g}")
    p2048 = pd.read_csv(f"{RES}/{sorted(f for f in os.listdir(RES) if f.startswith('pareto2048_') and f.endswith('_summary.csv'))[-1]}")
    sp = p2048.groupby(["dtype", "frac", "cap"]).speedup_pct.mean()
    v = pd.concat([runs(latest("gptol2048_")), runs(latest("gptolemu2048_"))])
    emu_cal = pd.concat([runs(f[:-4]) for f in sorted(os.listdir(LOG))
                         if f.startswith("gpsel") and f.endswith(".out") and not f.startswith("gpsel2048")])
    kappa_e = emu_cal[emu_cal.variant == "e12"].dll.max() / err[("g8", 12)]
    print("kappa_e (emulation, 12 moduli, gpsel without block 2048)", f"{kappa_e:.3g}")
    rows = []
    for scheme, tau in [(sch, t) for sch in ("shared", "per-method") for t in TAUS]:
        eps = tau / kappa
        eps_e = eps if scheme == "shared" else tau / kappa_e
        bits = min(b for (dt, b), e in err.items() if dt == "fp64" and b < 52 and e <= eps)
        mods = min(M for (dt, M), e in err.items() if dt == "g8" and e <= eps_e)
        tv, ev = f"t{bits}n", f"e{mods}"
        for cap, g in v.groupby("cap"):
            pred_t, pred_e = sp[("fp64", bits, cap)], sp[("g8", mods, cap)]
            # paired within the same job and round
            meas = {x: np.mean([100 * (h[h.variant == "native"].wall.mean() / h[h.variant == x].wall.mean() - 1)
                                for _, h in g.groupby(["job", "round"]) if (h.variant == x).any()]) for x in (tv, ev)}
            rows.append(dict(scheme=scheme, tau=tau, eps=eps, eps_e=eps_e, bits=bits, moduli=mods, cap=cap,
                             err_t=err[("fp64", bits)], err_e=err[("g8", mods)],
                             dll_t=g[g.variant == tv].dll.max(), dll_e=g[g.variant == ev].dll.max(),
                             ok_t=g[g.variant == tv].ok.all(), ok_e=g[g.variant == ev].ok.all(),
                             pred_t=pred_t, pred_e=pred_e, meas_t=meas[tv], meas_e=meas[ev],
                             pick="emu" if pred_e > pred_t else "trunc", best="emu" if meas[ev] > meas[tv] else "trunc"))
    d = pd.DataFrame(rows)
    d["within_tau"] = (d.dll_t <= d.tau) & (d.dll_e <= d.tau)
    d["correct"] = d.pick == d.best
    d.to_csv(f"{RES}/gptol.csv", index=False)
    pd.set_option("display.width", 220)
    print(d.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
