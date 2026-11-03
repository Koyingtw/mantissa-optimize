"""Energy of the Gaussian-process likelihood by phase (review round 9, minor 5): scripts/gpenergy.sh runs gp_chol with
GP_ENERGY=1 at a 1300 MHz lock and 1000 W, synchronizing around every trailing update and reading the hardware energy
counter there. Per variant (mean over 8 GPUs, 2 rounds and both passes): energy and time of the trailing GEMMs and of
everything else; dynamic power P - P0 (P0 = idle 262 W) of each phase; the in-program GEMM ratio
R_gemm = (P_gemm,v - P0)/(P_gemm,n - P0); the GEMM's share of the native dynamic energy w_E and of the run time phi; and
the whole-program ratio R_whole, which equals 1 - w_E (1 - R_gemm) only if the other phases do not change.
usage: gp_energy.py <run id> [ENERGY_GP|ENERGY_MP2] [out name]   → results/gp_energy.csv (or the given name)
Review round 10: the same lines come from DF-MP2 (DFMP2_ENERGY=1, ENERGY_MP2) and from runs that also had the 100 ms sampler
on (scripts/xval.sh), whose windows with equal native and truncated temperatures are used for both applications.
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P0 = 262.0
rid = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "ENERGY_GP"
OUT = sys.argv[3] if len(sys.argv) > 3 else "gp_energy.csv"
rows = []
for line in open(f"{HERE}/../logs/{rid}.out"):
    if line.startswith(TAG):
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        rows.append({k: (float(v) if k not in ("variant", "pass") else v) for k, v in kv.items()})
d = pd.DataFrame(rows)
g = d.groupby("variant")[["gemm_j", "gemm_s", "total_j", "wall"]].mean()
g["rest_j"] = g.total_j - g.gemm_j
g["rest_s"] = g.wall - g.gemm_s
g["P_gemm"] = g.gemm_j / g.gemm_s
g["P_rest"] = g.rest_j / g.rest_s
g["P_whole"] = g.total_j / g.wall
n = g.loc["native"]
g["R_gemm"] = (g.P_gemm - P0) / (n.P_gemm - P0)
g["R_rest"] = (g.P_rest - P0) / (n.P_rest - P0)
g["R_whole"] = (g.P_whole - P0) / (n.P_whole - P0)
g["phi"] = g.gemm_s / g.wall
g["w_E"] = (g.gemm_j - P0 * g.gemm_s) / (g.total_j - P0 * g.wall)
g["R_from_gemm"] = 1 - g.loc["native", "w_E"] * (1 - g.R_gemm)  # whole-program ratio if only the GEMM phases changed
g["t_gemm_ratio"] = g.gemm_s / n.gemm_s
g["t_rest_ratio"] = g.rest_s / n.rest_s
g.to_csv(f"{HERE}/../results/{OUT}")
pd.set_option("display.width", 220)
print(g[["P_gemm", "P_rest", "P_whole", "phi", "w_E", "R_gemm", "R_rest", "R_whole", "R_from_gemm", "t_gemm_ratio",
         "t_rest_ratio"]].round(4).to_string())
