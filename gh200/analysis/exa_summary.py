"""Summarise exageostat_tdp.sbatch jobs (real ExaGeoStatCPP MLE under module power caps, mantissa shim on the
FP64 tile GEMMs).

usage: exa_summary.py logs/exageostat/<name>_<jobid> [...]   (EXA_OUT=<csv> overrides results/exageostat_tdp.csv)
For every run: mean factorisation time over the likelihood evaluations after the first (the first one still
calibrates StarPU's performance models), its change relative to the native and to the copy-baseline run of the same
cap and pass, the module/GPU power and SM clock while the GPU factorises (GPU power above 200 W), and the largest
relative log-likelihood difference from the native run at the same evaluation points.
"""
import datetime
import glob
import os
import re
import sys

import pandas as pd

RES = re.compile(r"RESULT_EXA tdp=(\d+) variant=(\S+) pass=(\S) rc=(\d+) facto=\[([^\]]*)\] t_begin=([\d.]+) t_end=([\d.]+)")
LL = re.compile(r"(\d+) - Model Parameters \(([^)]*)\)----> LogLi: (\S+)")


def load_smi(path):
    out = []
    for line in open(path):
        f = [x.strip() for x in line.split(",")]
        try:
            t = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
            out.append((t.timestamp(), float(f[2]), float(f[3]), float(f[4]), float(f[8])))
        except (ValueError, IndexError):
            pass
    return pd.DataFrame(out, columns=["t", "mod_w", "gpu_w", "sm", "limit"])


def summarize(job_dir):
    smi = load_smi(os.path.join(job_dir, "smi.csv"))
    job_out = glob.glob(job_dir + ".out")[0]
    rows = []
    for m in RES.finditer(open(job_out).read()):
        tdp, var, pas, rc, facto, t0, t1 = m.groups()
        f = [float(x) for x in facto.split()]
        lls = {}
        out = os.path.join(job_dir, f"tdp{tdp}_{var}_{pas}.out")
        if os.path.exists(out):
            for k, th, ll in LL.findall(open(out).read()):
                lls[(int(k), th)] = float(ll)
        w = smi[(smi.t >= float(t0)) & (smi.t <= float(t1))]
        busy = w[w.gpu_w > 200]  # GPU factorising (idle draws 100-140 W)
        rows.append(dict(job=os.path.basename(job_dir), tdp=int(tdp), variant=var, pass_=pas, rc=int(rc),
                         evals=len(f), facto_s=sum(f[1:]) / max(len(f) - 1, 1), facto_first_s=f[0] if f else float("nan"),
                         busy_s=len(busy) * 0.25, mod_w=busy.mod_w.mean(), gpu_w=busy.gpu_w.mean(), sm_mhz=busy.sm.mean(),
                         lim=w.limit.min(), limx=w.limit.max(), lls=lls))
    d = pd.DataFrame(rows)
    for (tdp, pas), g in d.groupby(["tdp", "pass_"]):
        for ref_name, col in (("native", "vs_native_pct"), ("copy", "vs_copy_pct")):
            ref = g[g.variant == ref_name]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            d.loc[g.index, col] = (ref.facto_s / g.facto_s - 1) * 100
    # log-likelihood: relative difference from the first native run at the same (evaluation, theta)
    nat = d[d.variant == "native"]
    ref_ll = nat.iloc[0].lls if not nat.empty else {}
    d["dll_rel_max"] = [max((abs(v / ref_ll[k] - 1) for k, v in r.items() if k in ref_ll), default=float("nan"))
                        for r in d.lls]
    return d.drop(columns="lls")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = pd.concat([summarize(p.rstrip("/")) for p in sys.argv[1:]], ignore_index=True)
    os.makedirs("results", exist_ok=True)
    d.to_csv(os.environ.get("EXA_OUT", "results/exageostat_tdp.csv"), index=False)
    cols = ["tdp", "variant", "pass_", "rc", "evals", "facto_s", "vs_native_pct", "vs_copy_pct", "mod_w", "gpu_w",
            "sm_mhz", "busy_s", "dll_rel_max", "lim", "limx"]
    print(d[[c for c in cols if c in d]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n平均（兩輪）：")
    print(d.groupby(["tdp", "variant"])[["facto_s", "vs_native_pct", "vs_copy_pct", "sm_mhz", "dll_rel_max"]].mean()
          .to_string(float_format=lambda x: f"{x:.4g}"))
