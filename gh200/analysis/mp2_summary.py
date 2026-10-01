"""Summarise mp2_tdp.sbatch jobs (GPU4PySCF DF-MP2 variants under module power caps).

usage: mp2_summary.py logs/mp2/<name>_<jobid> [...]   (MP2_OUT=<csv> overrides results/mp2_summary.csv)

Every RESULT_MP2 line is one DF-MP2 run after a shared DF-RHF; its window is joined with the job's 250 ms
nvidia-smi log. Speed and energy are relative to the native run of the same cap and pass (ABBA pairs);
dE is the MP2 correlation energy minus the first native run's (Eh).
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi, window_stats  # noqa: E402

LINE = re.compile(r"RESULT_MP2 tdp=(\S+) variant=(\S+) pass=(\S) Ecorr=(\S+) dE=(\S+) wall=([\d.]+)s "
                  r"t_begin=([\d.]+) t_end=([\d.]+)")


def summarize(job_dir):
    smi = load_smi(os.path.join(job_dir, "smi.csv"))
    rows = []
    for f in sorted(glob.glob(os.path.join(job_dir, "tdp*.out"))):
        for m in LINE.finditer(open(f).read()):
            tdp, var, pas, ecorr, de, wall, t0, t1 = m.groups()
            w = window_stats(smi, float(t0), float(t1), busy_gpu_w=150)
            rows.append(dict(job=os.path.basename(job_dir), tdp=int(tdp), variant=var, pass_=pas, ecorr=float(ecorr),
                             dE=float(de), wall_s=float(wall), mod_w=w["mod_w"], sm_mhz=w["sm_mhz"],
                             energy_kj=w["energy_kj"], powercap=w["powercap_frac"], lim=w["limit_min"], limx=w["limit_max"]))
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
    d.to_csv(os.environ.get("MP2_OUT", "results/mp2_summary.csv"), index=False)
    cols = [c for c in ["job", "tdp", "variant", "pass_", "wall_s", "speedup_pct", "mod_w", "dmod_w", "sm_mhz", "energy_kj",
                        "energy_pct", "dE", "powercap", "lim", "limx"] if c in d]
    print(d[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n平均（各 ABBA 輪）：")
    print(d.groupby(["tdp", "variant"])[["speedup_pct", "energy_pct", "dmod_w", "dE"]].mean()
          .to_string(float_format=lambda x: f"{x:.3g}"))
