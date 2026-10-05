"""Summarise gp_tdp.sbatch jobs (GP log-likelihood with blocked Cholesky under module power caps).

usage: gp_summary.py logs/gp/<name>_<jobid> [...]   (GP_OUT=<csv> overrides results/gp_summary.csv)
Each RESULT_GP line is one variant (all range parameters); speed and energy are relative to the native
run of the same cap and pass; dll_max is the largest |log-likelihood difference| from the first native run.
RESULT_MLE lines (GP_MODE=mle, a whole maximum-likelihood fit per variant) are read the same way and add the
fitted range theta, its relative change from native and the number of likelihood evaluations.
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi, window_stats  # noqa: E402

LINE = re.compile(r"RESULT_GP tdp=(\S+) variant=(\S+) pass=(\S) status=(\S+) loglik=\[([^\]]*)\] dll=\[([^\]]*)\] "
                  r"wall=([\d.]+)s t_begin=([\d.]+) t_end=([\d.]+)")
MLE = re.compile(r"RESULT_MLE tdp=(\S+) variant=(\S+) pass=(\S) theta=(\S+) loglik=(\S+) dtheta_rel=(\S+) dll=(\S+) "
                 r"evals=(\d+) wall=([\d.]+)s t_begin=([\d.]+) t_end=([\d.]+)")


def summarize(job_dir):
    smi = load_smi(os.path.join(job_dir, "smi.csv"))
    rows = []
    for f in sorted(glob.glob(os.path.join(job_dir, "tdp*.out"))):
        text = open(f).read()
        for m in LINE.finditer(text):
            tdp, var, pas, status, ll, dll, wall, t0, t1 = m.groups()
            d = [abs(float(x)) for x in dll.split()]
            w = window_stats(smi, float(t0), float(t1), busy_gpu_w=150)
            rows.append(dict(job=os.path.basename(job_dir), tdp=int(tdp), variant=var, pass_=pas, status=status,
                             dll_max=max(d) if d else float("nan"), wall_s=float(wall), mod_w=w["mod_w"],
                             sm_mhz=w["sm_mhz"], energy_kj=w["energy_kj"], powercap=w["powercap_frac"],
                             lim=w["limit_min"], limx=w["limit_max"]))
        for m in MLE.finditer(text):
            tdp, var, pas, theta, ll, dth, dll, evals, wall, t0, t1 = m.groups()
            w = window_stats(smi, float(t0), float(t1), busy_gpu_w=150)
            rows.append(dict(job=os.path.basename(job_dir), tdp=int(tdp), variant=var, pass_=pas, status="mle",
                             theta=float(theta), dtheta_rel=float(dth), evals=int(evals), dll_max=abs(float(dll)),
                             wall_s=float(wall), mod_w=w["mod_w"], sm_mhz=w["sm_mhz"], energy_kj=w["energy_kj"],
                             powercap=w["powercap_frac"], lim=w["limit_min"], limx=w["limit_max"]))
    d = pd.DataFrame(rows)
    for (tdp, pas), g in d.groupby(["tdp", "pass_"]):
        ref = g[g.variant == "native"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        d.loc[g.index, "speedup_pct"] = (ref.wall_s / g.wall_s - 1) * 100
        d.loc[g.index, "energy_pct"] = (g.energy_kj / ref.energy_kj - 1) * 100
        d.loc[g.index, "dmod_w"] = g.mod_w - ref.mod_w
    return d


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = pd.concat([summarize(p.rstrip("/")) for p in sys.argv[1:]], ignore_index=True)
    os.makedirs("results", exist_ok=True)
    d.to_csv(os.environ.get("GP_OUT", "results/gp_summary.csv"), index=False)
    cols = ["tdp", "variant", "pass_", "status", "wall_s", "speedup_pct", "mod_w", "dmod_w", "sm_mhz",
            "energy_pct", "dll_max", "theta", "dtheta_rel", "evals", "powercap", "lim", "limx"]
    print(d[[c for c in cols if c in d]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n平均（各 ABBA 輪）：")
    print(d.groupby(["tdp", "variant"])[["speedup_pct", "energy_pct", "dmod_w", "dll_max"]].mean()
          .to_string(float_format=lambda x: f"{x:.3g}"))
