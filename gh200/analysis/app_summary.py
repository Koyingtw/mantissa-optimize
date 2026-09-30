"""Summarise app_tdp.sbatch jobs (MFEM, SW4, LAMMPS): application speed, correctness and power per variant.

usage: app_summary.py logs/apps/<name>_<jobid> [...]   (APP_OUT=<csv> overrides results/app_summary.csv)
Speed is the application's own timer (MFEM: ms per CG iteration; SW4: solver seconds; LAMMPS: loop time),
so setup and I/O do not dilute it; energy is the whole run window.  Ratios are against the native run
at the same cap and ABBA pass.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import RUN_RE, load_smi, window_stats  # noqa: E402


def parse_output(path):
    if not os.path.exists(path):
        return {}
    t = open(path, errors="replace").read()
    r = {}
    m = re.search(r"\[TIME\] solves=\d+ total_its=(\d+) solve_time=([\d.]+) s\s+per_it=([\d.]+) ms", t)
    if m:  # MFEM
        r.update(its=int(m.group(1)), solve_s=float(m.group(2)), app_time=float(m.group(3)))
        m = re.search(r"\(B,AB\)=([-\d.e+]+)", t)
        r["check"] = float(m.group(1)) if m else None
        m = re.search(r"\|\|X\|\|=([-\d.e+]+)", t)
        r["result"] = float(m.group(1)) if m else None
        m = re.search(r"nodal_err_max=([-\d.e+]+)", t)
        r["err"] = float(m.group(1)) if m else None
        return r
    m = re.search(r"Loop time of ([\d.]+) on", t)
    if m:  # LAMMPS
        r["app_time"] = float(m.group(1))
        e = re.findall(r"^\s+\d+\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)", t, re.M)
        if e:
            r["result"] = float(e[-1][2])  # last thermo line, third column (total energy in the kk inputs)
        return r
    # SW4 prints e.g. "Execution time, solver phase 1 minute  40.94 seconds" or "... 9.87e+01 seconds"
    m = re.search(r"solver phase\s+(?:(\d+)\s+hours?\s+)?(?:(\d+)\s+minutes?\s+)?([\d.eE+-]+)\s+seconds", t)
    if m:  # SW4
        r["app_time"] = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
    return r


def summarize(job_dir):
    log = job_dir + ".out"
    smi = load_smi(os.path.join(job_dir, "smi.csv"))
    rows = []
    for tdp, var, pas, t0, t1 in RUN_RE.findall(open(log).read()):
        tag = f"tdp{tdp}_{var.replace(':', '_')}" + ("_b" if pas == "B" else "")
        r = {"job": os.path.basename(job_dir), "tdp": int(tdp), "variant": var, "pass": pas,
             "wall_s": float(t1) - float(t0)}
        r.update(parse_output(os.path.join(job_dir, tag + ".out")))
        w = window_stats(smi, float(t0), float(t1), busy_gpu_w=150)
        r.update(mod_w=w["mod_w"], sm_mhz=w["sm_mhz"], energy_kj=w["energy_kj"], powercap=w["powercap_frac"],
                 lim=w["limit_min"], limx=w["limit_max"])
        rows.append(r)
    d = pd.DataFrame(rows)
    for (tdp, pas), g in d.groupby(["tdp", "pass"]):
        ref = g[g.variant == "native"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        d.loc[g.index, "speedup_pct"] = (ref.app_time / g.app_time - 1) * 100
        d.loc[g.index, "energy_pct"] = (g.energy_kj / ref.energy_kj - 1) * 100
        d.loc[g.index, "dmod_w"] = g.mod_w - ref.mod_w
        if "result" in d:
            d.loc[g.index, "result_rel"] = (g.result - ref.result).abs() / abs(ref.result)
    return d


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = pd.concat([summarize(p.rstrip("/")) for p in sys.argv[1:]], ignore_index=True)
    os.makedirs("results", exist_ok=True)
    d.to_csv(os.environ.get("APP_OUT", "results/app_summary.csv"), index=False)
    cols = [c for c in ["job", "tdp", "variant", "pass", "app_time", "its", "speedup_pct", "mod_w", "dmod_w", "sm_mhz",
                        "energy_kj", "energy_pct", "powercap", "result_rel", "err", "lim", "limx"] if c in d]
    print(d[cols].to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    agg = d.groupby(["tdp", "variant"])[["speedup_pct", "energy_pct", "dmod_w"]].mean()
    print("\nmean of the ABBA passes:")
    print(agg.to_string(float_format=lambda x: f"{x:.2f}"))
