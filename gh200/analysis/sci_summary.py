"""Summarise sci_sweep.sbatch jobs: result error, iterations, time and power per variant.

usage: sci_summary.py logs/sci/<name>_<jobid> [...]  > table
The reference for the error is the 'stats' run (unmodified GEMMs) at the same TDP.
QE energies are in Ry, PySCF in Hartree (1 Ha = 627.5 kcal/mol; chemical accuracy 1.6e-3 Ha).
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi, window_stats  # noqa: E402

RUN = re.compile(r"=== END tdp=(\d+) variant=(\S+) pass=(\S) t_begin=([\d.]+) t_end=([\d.]+)")


def parse_output(path):
    t = open(path, errors="replace").read()
    r = {}
    m = re.findall(r"^!\s+total energy\s+=\s+([-\d.]+) Ry", t, re.M)
    if m:  # Quantum ESPRESSO
        r["energy"] = float(m[-1])
        it = re.search(r"convergence has been achieved in\s+(\d+) iterations", t)
        r["iters"] = int(it.group(1)) if it else None
        r["converged"] = bool(it)
        w = re.search(r"PWSCF\s+:.*CPU\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)s WALL", t)
        if w:
            r["app_wall_s"] = int(w.group(1) or 0) * 3600 + int(w.group(2) or 0) * 60 + float(w.group(3))
        return r
    m = re.search(r"RESULT converged=(\w+) E=([-\d.]+) Eh\s+scf_cycles=(\d+)\s+wall=([\d.]+)s", t)
    if m:  # PySCF DFT
        return {"energy": float(m.group(2)), "converged": m.group(1) == "True", "iters": int(m.group(3)),
                "app_wall_s": float(m.group(4))}
    hf = re.search(r"RESULT_HF converged=(\w+) E=([-\d.]+) Eh wall=([\d.]+)s", t)
    mp = re.search(r"RESULT_MP2 Ecorr=([-\d.]+) Etot=([-\d.]+) Eh wall=([\d.]+)s", t)
    if hf and mp:  # PySCF HF + MP2
        return {"energy": float(mp.group(2)), "e_hf": float(hf.group(2)), "e_corr": float(mp.group(1)),
                "converged": hf.group(1) == "True", "app_wall_s": float(hf.group(3)) + float(mp.group(3))}
    return {"converged": False}


def summarize(job_dir):
    log = job_dir + ".out"
    smi = load_smi(os.path.join(job_dir, "smi.csv"))
    case_dirs = [d for d in glob.glob(os.path.join(job_dir, "*")) if os.path.isdir(d)]
    rows = []
    for tdp, var, pas, t0, t1 in RUN.findall(open(log).read()):
        tag = f"tdp{tdp}_{var.replace(':', '_')}" + ("" if pas == "A" else "_" + pas.lower())
        out = next((os.path.join(d, tag + ".out") for d in case_dirs if os.path.exists(os.path.join(d, tag + ".out"))),
                   None)
        r = {"job": os.path.basename(job_dir), "tdp": int(tdp), "variant": var, "pass": pas,
             "wall_s": float(t1) - float(t0)}
        r.update(parse_output(out) if out else {"converged": False})
        w = window_stats(smi, float(t0), float(t1), busy_gpu_w=150)
        r.update(mod_w=w["mod_w_all"], sm_mhz=w["sm_mhz"], energy_kj=w["energy_kj"], powercap=w["powercap_frac"])
        rows.append(r)
    d = pd.DataFrame(rows)
    for tdp, g in d.groupby("tdp"):
        ref = g[g.variant == "stats"]
        if ref.empty:  # jobs without a statistics run: the unmodified application is the reference
            ref = g[g.variant == "native"]
        if not ref.empty and "energy" in d:
            d.loc[g.index, "dE"] = g.energy - ref.energy.iloc[0]
        cp = g[g.variant == "copy"]  # both ABBA passes averaged
        if not cp.empty:
            d.loc[g.index, "time_vs_copy"] = g.wall_s / cp.wall_s.mean()
            d.loc[g.index, "energy_vs_copy"] = g.energy_kj / cp.energy_kj.mean()
    return d


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    frames = [summarize(p.rstrip("/")) for p in sys.argv[1:]]
    d = pd.concat(frames, ignore_index=True)
    os.makedirs("results", exist_ok=True)
    d.to_csv(os.environ.get("SCI_OUT", "results/sci_summary.csv"), index=False)
    cols = [c for c in ["job", "tdp", "variant", "pass", "energy", "dE", "iters", "converged", "wall_s", "app_wall_s",
                        "time_vs_copy", "mod_w", "sm_mhz", "energy_kj", "energy_vs_copy", "powercap"] if c in d]
    print(d[cols].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
