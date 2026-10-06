"""Helpers to join nvidia-smi power logs with benchmark run windows."""
import glob
import os
import re

import pandas as pd

COLS = ["ts", "mod_inst", "mod_avg", "gpu_inst", "sm", "mem", "temp", "reason", "limit"]
THERMAL = 0x20 | 0x40  # sw / hw thermal slowdown
POWER_CAP = 0x04


def load_smi(path):
    """Load one CSV, or every CSV in a directory (one per host)."""
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    frames = []
    for f in files:
        d = pd.read_csv(f, header=None, names=COLS, skipinitialspace=True, dtype=str)
        for c in COLS[1:8]:
            if c != "reason":
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d["reason"] = d["reason"].map(lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else 0)
        # nvidia-smi prints node-local time; the cluster runs in UTC.
        d["t"] = (pd.to_datetime(d["ts"], format="%Y/%m/%d %H:%M:%S.%f", utc=True) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
        d["host"] = os.path.basename(f)[:-4]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def window_stats(smi, t0, t1, busy_gpu_w=300):
    """Mean power/clock over [t0, t1] restricted to busy samples, averaged per host."""
    w = smi[(smi.t >= t0) & (smi.t <= t1)]
    busy = w[w.gpu_inst > busy_gpu_w]
    if busy.empty:
        busy = w
    return {
        "mod_w": busy.mod_inst.mean(),
        "mod_w_all": w.mod_inst.mean(),
        "gpu_w": busy.gpu_inst.mean(),
        "sm_mhz": busy.sm.mean(),
        "temp_max": w.temp.max(),
        "temp_start": smi[(smi.t >= t0 - 3) & (smi.t <= t0 + 1)].temp.mean(),
        "thermal_frac": (busy.reason & THERMAL > 0).mean(),
        "powercap_frac": (busy.reason & POWER_CAP > 0).mean(),
        # Energy = mean module power over the whole window x duration, summed over hosts.
        # Uses the driver's 1-s averaged module power (the value the power limiter acts
        # on): same mean as the instantaneous reading for steady work, but it does not
        # alias with bursty kernels (cuBLAS FP64 emulation read -37..+41% off with the
        # instantaneous sensor).  energy_kj_inst keeps the old estimate for comparison.
        "energy_kj": w.groupby("host").mod_avg.mean().sum() * (t1 - t0) / 1e3,
        "energy_kj_inst": w.groupby("host").mod_inst.mean().sum() * (t1 - t0) / 1e3,
        "mod_w_avg": busy.mod_avg.mean(),
        # the module power limit actually in force during the window (other users'
        # tools can reset caps cluster-wide); runs whose limit differs from the
        # intended TDP must be discarded
        "limit_min": pd.to_numeric(w["limit"], errors="coerce").min(),
        "limit_max": pd.to_numeric(w["limit"], errors="coerce").max(),
    }


RUN_RE = re.compile(r"=== END tdp=(\d+) variant=(\S+) pass=(\S) t_begin=([\d.]+) t_end=([\d.]+)")


def parse_hpl_log(path):
    """Split a hpl_tdp.sbatch log into runs with score, residual and pass/fail."""
    text = open(path).read()
    rows = []
    for block in text.split("=== RUN ")[1:]:
        end = RUN_RE.search(block)
        if not end:
            continue
        tdp, var, pas, t0, t1 = end.groups()
        if pas == "W":  # uncounted warm-up run (hpl_tdp.sbatch WARMUP=1)
            continue
        row = {"tdp": int(tdp), "variant": var, "pass": pas, "t0": float(t0), "t1": float(t1)}
        m = re.search(r"Final Score:\s+([\d.e+]+)\s+GFLOPS", block)
        if m:  # rocHPL
            row["tflops"] = float(m.group(1)) / 1e3
            row["residual"] = float(re.search(r"Residual value = ([\d.e+-]+)", block).group(1))
            row["passed"] = "Residual Check: PASSED" in block
        m = re.search(r"GFLOPS = ([\d.e+]+), per GPU", block)
        if m:  # HPL-MxP
            clean = re.sub(r"\x1b\[[0-9;]*m", "", block)
            row["tflops"] = float(re.search(r"\s GFLOPS = ([\d.e+]+), per GPU", clean).group(1)) / 1e3
            row["lu_tflops"] = float(re.search(r"LU GFLOPS = ([\d.e+]+)", clean).group(1)) / 1e3
            res = re.findall(r"\* N\) =\s+([\d.E+-]+) \.+ (PASSED|FAILED)", clean)
            row["residual"] = float(res[-1][0]) if res else float("nan")
            row["passed"] = bool(res) and res[-1][1] == "PASSED"
            row["iters"] = len(re.findall(r"Solver iteration", clean))
        rows.append(row)
    return pd.DataFrame(rows)
