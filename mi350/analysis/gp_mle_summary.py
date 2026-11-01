"""Full maximum-likelihood fit of the Gaussian process on MI350X (scripts/gp_mle.sh → RESULT_MLE lines): relative change
of each estimate (range, nugget, variance) of a truncated variant from the native fit on the same data set (seed).
usage: gp_mle_summary.py <run id>   → results/<run id>_mle.csv
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rid = sys.argv[1]
rows = []
for line in open(f"{HERE}/../logs/{rid}.out"):
    if line.startswith("RESULT_MLE"):
        kv = dict(re.findall(r"(\w+)=([^\s]+)", line.replace("tag=seed=", "seed=")))
        rows.append(dict(seed=int(kv["seed"]), variant=kv["variant"], range=float(kv["range"]), nugget=float(kv["nugget"]),
                         sigma2=float(kv["sigma2"]), maxll=float(kv["maxll"]), evals=int(kv["evals"])))
d = pd.DataFrame(rows).sort_values(["seed", "variant"])
nat = d[d.variant == "native"].set_index("seed")
for p in ("range", "nugget", "sigma2"):
    d[f"d{p}_pct"] = 100 * (d[p] / nat[p].reindex(d.seed).values - 1)
d["dmaxll"] = d.maxll - nat.maxll.reindex(d.seed).values
d.to_csv(f"{HERE}/../results/{rid}_mle.csv", index=False)
pd.set_option("display.width", 200)
print(d.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
