"""MI350X application gains with a mixed-effects model instead of t intervals over analysis units (review round 5, M6).

HPL / HPL-MxP: log(TFLOP/s) ~ variant + node, random intercept per ABBA block (runs in one block share the thermal
and job state). GP / DF-MP2: log(1/wall) per GPU and round ~ variant + node + GPU (fixed, 16 levels), random intercept
per (node, round). The variant effect is reported as a percentage with its 95% Wald interval, next to the pooled t
interval of results/apps_units.csv.
usage: mixed_effects.py → results/mixed_effects.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from apps_units import APPS, RES  # noqa: E402

warnings.simplefilter("ignore")


def load(rid, node, kind, var, nat):
    d = pd.read_csv(f"{RES}/{rid}_raw.csv")
    d["variant"] = d.variant.astype(str)
    d = d[d.variant.isin([var, nat])].copy()
    if kind == "run":
        d = d[d.passed & (d.pass_ != "W")]
        d["block"] = d["block"] if "block" in d else d.pass_.str[1:]
        d["y"] = np.log(d.tflops)
        d["grp"] = node + ":" + d.block.astype(str)
    else:
        if "cold" in d:
            d = d[~d.cold]
        d["y"] = -np.log(d.wall)
        d["grp"] = node + ":" + d["round"].astype(str)
        d["gpu"] = node + ":" + d.gpu.astype(str)
    d["node"] = node
    d["trunc"] = (d.variant == var).astype(float)
    return d


def main():
    mp2n02 = sorted(f[:-8] for f in os.listdir(RES) if f.startswith("mp2n02_") and f.endswith("_raw.csv"))
    units = pd.read_csv(f"{RES}/apps_units.csv")
    rows = []
    for app, (var, nat, nodes) in APPS.items():
        parts = []
        for node, rid, _, kind in nodes:
            if rid == "MP2N02":
                rid = mp2n02[-1]
            parts.append(load(rid, node, kind, var, nat))
        d = pd.concat(parts)
        kind = nodes[0][3]
        for cap, g in d.groupby("cap"):
            form = "y ~ trunc + C(node)" if kind == "run" else "y ~ trunc + C(gpu)"
            m = smf.mixedlm(form, g, groups=g["grp"]).fit(reml=True)
            b, se = m.params["trunc"], m.bse["trunc"]
            u = units[(units.app == app) & (units.cap == cap)]
            rows.append(dict(app=app, cap=cap, groups=g.grp.nunique(), obs=len(g),
                             me_pct=100 * (np.exp(b) - 1), me_lo=100 * (np.exp(b - 1.96 * se) - 1),
                             me_hi=100 * (np.exp(b + 1.96 * se) - 1),
                             t_mean=float(u["mean"].iloc[0]) if len(u) else np.nan,
                             t_ci=float(u["ci95"].iloc[0]) if len(u) else np.nan))
    r = pd.DataFrame(rows).sort_values(["app", "cap"], ascending=[True, False])
    r["me_halfwidth"] = (r.me_hi - r.me_lo) / 2
    r.to_csv(f"{RES}/mixed_effects.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
