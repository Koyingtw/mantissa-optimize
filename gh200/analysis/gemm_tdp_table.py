"""Per (TDP, dtype, frac) table of throughput, 1-s averaged module power and SM clock for any
e1_gemm_tdp.sbatch job (gemm_mant_bench, fp64_kernel_bench, attn_bench), passes averaged.
usage: gemm_tdp_table.py JOB [JOB...]   gain_pct is vs that dtype's native width at the same TDP."""
import re
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from smi import load_smi, window_stats  # noqa: E402

NATIVE = {"fp64": 52, "bf16": 7, "fp8": 3, "fp32": 23, "tf32": 23, "dfma": 52, "nbody": 52, "copy": 52, "triad": 52, "cg": 52, "tf32x3": 23, "gemmul8": 99, "tf32x1": 23, "tfp32": 23, "attn": 7, "rbf16": 7, "rfp8": 3}
rows = []
for j in sys.argv[1:]:
    smi = load_smi(f"logs/e1_{j}_smi.csv")
    for line in open(f"logs/e1_{j}.out"):
        if not line.startswith("RESULT"):
            continue
        kv = dict(re.findall(r"(\w+)=([^\s,]+)", line.replace("tag=", "")))
        w = window_stats(smi, float(kv["t_begin"]), float(kv["t_end"]))
        rows.append(dict(job=j, tdp=int(kv["tdp"]), dtype=kv["dtype"], frac=int(kv["frac"]), low=kv.get("low", "zero"),
                         inner=int(kv["n"]) if kv["dtype"] == "nbody" else -1, tflops=float(kv["tflops"]),
                         err=float(kv.get("rel_err", -1)), W=w["mod_w_avg"], MHz=w["sm_mhz"],
                         lim=w["limit_min"], limx=w["limit_max"]))
d = pd.DataFrame(rows)
g = d.groupby(["job", "tdp", "dtype", "frac", "low", "inner"], sort=False).agg(
    tflops=("tflops", "mean"), W=("W", "mean"), MHz=("MHz", "mean"), err=("err", "max"), n=("tflops", "size"),
    lim=("lim", "min"), limx=("limx", "max")).reset_index()
nat = g[(g.frac == g.dtype.map(NATIVE)) & (g.low == "zero") & (g.inner.isin([-1, 52]))][["job", "tdp", "dtype", "tflops", "W"]]
g = g.merge(nat, on=["job", "tdp", "dtype"], suffixes=("", "_nat"), how="left")
g["gain_pct"] = (g.tflops / g.tflops_nat - 1) * 100
g["dW"] = g.W - g.W_nat
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print(g.drop(columns=["tflops_nat", "W_nat"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
