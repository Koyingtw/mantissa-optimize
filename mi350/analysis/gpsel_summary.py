"""GP likelihood with block size nb (k of the trailing updates) on MI350X: native, truncation (t35n) and GEMMul8 3.5.4
through its direct API (e12), per cap; held-out check of the per-call selector (review round 5, M4).
Each round runs "native t35n e12 e12 t35n native" in one process per GPU; speedup = mean native wall / mean variant wall
over the 8 GPUs of a round (both passes), interval over rounds; |dll| = largest log-likelihood deviation from native.
usage: gpsel_summary.py <run id> [...]  → results/gpsel_summary.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for rid in sys.argv[1:]:
    nb = int(re.search(r"gpsel(\d+)_", rid).group(1))
    for line in open(f"{HERE}/../logs/{rid}.out"):
        if line.startswith("RESULT_GP"):
            kv = dict(re.findall(r"(\w+)=([^\s,\[]+)", line.replace("tag=", "")))
            dll = [abs(float(x)) for x in re.search(r"dll=\[([^\]]*)\]", line).group(1).split()]
            rows.append(dict(nb=nb, cap=int(kv["cap"]), round=int(kv["round"]), gpu=int(kv["gpu"]), variant=kv["variant"],
                             pass_=kv["pass"], ok=kv["status"] == "ok", wall=float(kv["wall"].rstrip("s")),
                             dll=max(dll) if dll else np.nan))
d = pd.DataFrame(rows)
out = []
for (nb, cap), g in d.groupby(["nb", "cap"]):
    for v in ("t35n", "e12"):
        sp = [100 * (h[h.variant == "native"].wall.mean() / h[h.variant == v].wall.mean() - 1) for _, h in g.groupby("round")]
        gv = g[g.variant == v]
        out.append(dict(nb=nb, cap=cap, variant=v, rounds=len(sp), speedup_pct=np.mean(sp), lo=min(sp), hi=max(sp),
                        ok=gv.ok.all(), dll_max=gv.dll.max(), wall_native=g[g.variant == "native"].wall.mean()))
s = pd.DataFrame(out).sort_values(["nb", "cap", "variant"], ascending=[True, False, True])
s.to_csv(f"{HERE}/../results/gpsel_summary.csv", index=False)
pd.set_option("display.width", 200)
print(s.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
