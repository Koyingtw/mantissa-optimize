"""W4: decode-like weight streaming (scripts/ai_decode.sh): throughput, power and energy per token by format.

Every window of every GPU is joined with the power log by BDF (hardware energy accumulator, mi350/analysis/smi.py).
Within a cap and batch each format runs twice (ABBA); the reduced formats are compared with the native format of the
same family (bf16:7 for BF16, fp8:3 for FP8) on the same GPU, averaging the two windows of each format, then over GPUs
(mean and 95% t interval). Energy per token = window energy / tokens of that window.
usage: decode_summary.py <run id>   (logs in ../logs) → results/ai_decode.csv, prints changes vs native
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from e1_summary import BDF, ci  # noqa: E402
from smi import load_smi, window_stats  # noqa: E402

NATIVE = {"bf16": "bf16:7", "fp8": "fp8:3"}


def main():
    rid = sys.argv[1]
    smi = load_smi(f"{LOG}/{rid}_smi.csv")
    rows = []
    for f in sorted(glob.glob(f"{LOG}/{rid}_cap*_gpu*.log")):
        cap = int(re.search(r"_cap(\d+)_", f).group(1))
        for line in open(f):
            if line.startswith("RESULT"):
                r = json.loads(line[7:])
                w = window_stats(smi, BDF.index(r["bdf"]), r["t_begin"], r["t_end"])
                tok = r["tokens_per_s"] * (r["t_end"] - r["t_begin"])
                rows.append(dict(cap=cap, bdf=r["bdf"], window=r["window"], fmt=r["fmt"], batch=r["batch"],
                                 tokens_per_s=r["tokens_per_s"], power_w=w["power_acc_w"], gfxclk=w["gfxclk"],
                                 j_per_token=w["energy_j"] / tok, ppt_frac=w.get("ppt_frac", np.nan)))
    d = pd.DataFrame(rows)
    d["family"] = d.fmt.str.split(":").str[0]
    g = d.groupby(["cap", "batch", "bdf", "fmt", "family"], as_index=False)[
        ["tokens_per_s", "power_w", "gfxclk", "j_per_token", "ppt_frac"]].mean()
    out = []
    for (cap, batch, fam), h in g.groupby(["cap", "batch", "family"]):
        nat = h[h.fmt == NATIVE[fam]].set_index("bdf")
        for fmt, x in h.groupby("fmt"):
            x = x.set_index("bdf")
            n = nat.reindex(x.index)
            row = dict(cap=cap, batch=batch, fmt=fmt, gpus=len(x), tokens_per_s=x.tokens_per_s.mean(),
                       power_w=x.power_w.mean(), gfxclk=x.gfxclk.mean(), j_per_token=x.j_per_token.mean(),
                       ppt_frac=x.ppt_frac.mean())
            for col, name in (("tokens_per_s", "speed"), ("power_w", "power"), ("j_per_token", "energy")):
                r = x[col].values / n[col].values - 1
                lo, hi = ci(r)
                row[f"{name}_pct"], row[f"{name}_lo"], row[f"{name}_hi"] = 100 * r.mean(), 100 * lo, 100 * hi
            out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(HERE, "..", "results", "ai_decode.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["cap", "batch", "fmt", "tokens_per_s", "power_w", "gfxclk", "ppt_frac", "speed_pct", "power_pct", "energy_pct",
            "energy_lo", "energy_hi"]
    print(s[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
