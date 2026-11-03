"""Post-hoc FP64 spot check of GEMMul8 (review round 7, P3): parse the SPOT lines of scripts/spot.sh runs.
For each case and moduli count: span statistic of the span check, full and spot-estimated normwise error, their ratio,
and the spot check's time relative to the emulated call. A later run of the same case replaces an earlier one
(DF-MP2 was rerun sampling columns, the larger dimension).
usage: spot_summary.py → results/spot.csv
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
rows = {}
for f in sorted(x for x in os.listdir(LOG) if x.startswith("spot") and x.endswith(".out")):
    for line in open(f"{LOG}/{f}"):
        if line.startswith("SPOT"):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            kv["sample"] = "cols" if "cols" in kv else "rows"
            kv["run"] = f[:-4]
            rows[(kv["tag"], kv["moduli"])] = kv
d = pd.DataFrame(rows.values())
for c in ("m", "n", "k", "moduli"):
    d[c] = d[c].astype(int)
for c in ("span", "err_full", "err_spot", "ratio", "t_native_ms", "t_emu_ms", "t_spot_ms", "spot_over_emu"):
    d[c] = d[c].astype(float)
d = d[["tag", "m", "n", "k", "moduli", "sample", "span", "err_full", "err_spot", "ratio", "t_native_ms", "t_emu_ms",
       "t_spot_ms", "spot_over_emu", "run"]].sort_values(["tag", "moduli"])
d.to_csv(f"{HERE}/../results/spot.csv", index=False)
pd.set_option("display.width", 220)
print(d.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
