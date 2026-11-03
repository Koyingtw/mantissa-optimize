"""Post-hoc FP64 spot check of GEMMul8 (review rounds 7-8, P3/Q1): parse the SPOT (emulation) and TRUNC (truncation on
the same data) lines of scripts/spot*.sh runs. Per case, method setting and sampling: span statistics, full and
spot-estimated normwise error and their ratio, per-unit counts above 1e-10/1e-8/1e-6, componentwise error
max |dC|/(|A||B|) (full and on the samples, with counts above 1e-9/1e-6/1e-3 when present), and the spot check's time
relative to the call. A later run of the same case and setting replaces an earlier one.
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
        if line.startswith(("SPOT", "TRUNC")) and "status=" not in line:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            kv["kind"] = "emu" if line.startswith("SPOT") else "trunc"
            kv["setting"] = kv.get("moduli", kv.get("bits"))
            kv["sample"] = "cols" if "cols" in kv else "rows"
            kv["nsamp"] = kv.get("cols", kv.get("rows"))
            kv.setdefault("stratify", "0")
            kv["run"] = f[:-4]
            rows[(kv["tag"], kv["kind"], kv["setting"], kv["stratify"])] = kv
d = pd.DataFrame(rows.values())
cols = ["tag", "kind", "setting", "m", "n", "k", "sample", "nsamp", "stratify", "span", "span_b", "err_full", "err_spot",
        "ratio", "units", "bad10", "bad8", "bad6", "unit_max", "cw_full", "cw_spot", "cw9", "cw6", "cw3", "tiny",
        "t_native_ms", "t_emu_ms", "t_spot_ms", "spot_over_emu", "t_spot_cw_ms", "run"]
d = d.reindex(columns=cols)
d = d.sort_values(["tag", "kind", "setting", "stratify"])
d.to_csv(f"{HERE}/../results/spot.csv", index=False)
pd.set_option("display.width", 250)
print(d[["tag", "kind", "setting", "stratify", "span", "err_full", "ratio", "cw_full", "cw6", "spot_over_emu"]].to_string(index=False))
