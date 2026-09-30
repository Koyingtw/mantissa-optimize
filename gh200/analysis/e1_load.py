"""Parse E1 GEMM logs and join each measurement window with its power log."""
import re
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from smi import load_smi, window_stats  # noqa: E402


def load_e1(jobs, logdir="logs"):
    rows = []
    for j in jobs:
        smi = load_smi(f"{logdir}/e1_{j}_smi.csv")
        for line in open(f"{logdir}/e1_{j}.out"):
            if not line.startswith("RESULT"):
                continue
            kv = dict(re.findall(r"(\w+)=([^\s,]+)", line.replace("tag=", "")))
            w = window_stats(smi, float(kv["t_begin"]), float(kv["t_end"]))
            rows.append(dict(job=j, host=kv["host"], tdp=int(kv["tdp"]), pass_=kv["pass"], dtype=kv["dtype"],
                             frac=int(kv["frac"]), low=kv.get("low", "zero"), only_a=int(kv.get("only_a", 0)),
                             tflops=float(kv["tflops"]), **w))
    d = pd.DataFrame(rows)
    d["gflops_per_j"] = d.tflops * 1e3 / d.mod_w
    return d
