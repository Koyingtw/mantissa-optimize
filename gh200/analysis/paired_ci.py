"""Paired gains with 95% t intervals for repeated cap sweeps (HPL, HPL-MxP).

usage: paired_ci.py <bench> <native variant> <variant> <raw csv>[:label] [<raw csv>[:label] ...] [--out results/x.csv]
  bench    name written to the output (hpl, mxp, ...)
  raw csv  per-run output of summarize.py (<tag>_raw.csv: tdp, variant, pass, tflops, energy_kj, ...)

A pair is the native run and the variant run with the same cap and pass letter in the same job (ABBA blocks);
the gain is variant/native - 1 in throughput and 1 - variant/native in module energy per solve. Each job is
reported on its own and all jobs pooled (pairs from different node sets, so the pooled interval also contains
node-set differences).
"""
import argparse

import numpy as np
import pandas as pd

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}


def pairs(raw, native, variant, label):
    raw = raw.copy()
    raw["variant"] = raw.variant.astype(str)
    out = []
    for (tdp, pas), g in raw.groupby(["tdp", "pass"]):
        n = g[g.variant == native]
        v = g[g.variant == variant]
        if len(n) == 1 and len(v) == 1:
            out.append(dict(job=label, tdp=int(tdp), pass_=pas,
                            gain=(v.tflops.iloc[0] / n.tflops.iloc[0] - 1) * 100,
                            energy_saving=(1 - v.energy_kj.iloc[0] / n.energy_kj.iloc[0]) * 100))
    return pd.DataFrame(out)


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return x.mean() if n else np.nan, np.nan
    return x.mean(), T975[n - 1] * x.std(ddof=1) / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("native")
    ap.add_argument("variant")
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    allp = []
    for spec in a.raw:
        path, _, label = spec.partition(":")
        allp.append(pairs(pd.read_csv(path), a.native, a.variant, label or path))
    p = pd.concat(allp, ignore_index=True)
    rows = []
    for tdp, g in p.groupby("tdp"):
        for label, h in list(g.groupby("job")) + [("pooled", g)]:
            m, c = ci(h.gain)
            me, ce = ci(h.energy_saving)
            rows.append(dict(bench=a.bench, variant=a.variant, tdp=tdp, job=label, n=len(h), gain_mean=m,
                             gain_ci95=c, energy_saving_mean=me, energy_saving_ci95=ce,
                             gains=" ".join(f"{x:.2f}" for x in h.gain)))
    r = pd.DataFrame(rows).sort_values(["tdp", "job"], ascending=[False, True])
    print(r.round(2).to_string(index=False))
    if a.out:
        r.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
