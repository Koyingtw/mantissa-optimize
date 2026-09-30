"""95% confidence intervals for paired speedups (reduced vs native, same host, TDP and ABBA pass).

usage: ci.py e1  -> results/e1_ci.csv  (GEMM microbenchmark: 3 nodes x 2 passes = 6 pairs per cell)
Each pair is (reduced / native - 1) measured back to back on the same GPU, so node-to-node clock
spread cancels; the CI is mean +- t(0.975, n-1) * sd / sqrt(n).
"""
import sys

import pandas as pd

# two-sided 95% Student t quantiles by degrees of freedom (no scipy in the analysis venv)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df):
    return T975.get(df) or T975[max(k for k in T975 if k <= df)]


def paired_ci(d, keys, native_mask, value="tflops"):
    nat = d[native_mask].rename(columns={value: "native"})[keys + ["native"]]
    m = d[~native_mask].merge(nat, on=keys)
    m["gain"] = (m[value] / m.native - 1) * 100
    rows = []
    for g, x in m.groupby([c for c in m.columns if c in ("tdp", "dtype", "frac")]):
        n, mu, sd = len(x), x.gain.mean(), x.gain.std(ddof=1)
        h = t975(n - 1) * sd / n ** 0.5 if n > 1 else float("nan")
        rows.append(dict(zip(["tdp", "dtype", "frac"], g), n=n, gain_pct=mu, ci95=h, lo=mu - h, hi=mu + h))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if sys.argv[1] == "e1":
        d = pd.read_csv("results/e1_raw.csv")
        native = {"fp64": 52, "bf16": 7}
        mask = d.frac == d.dtype.map(native)
        r = paired_ci(d, ["host", "tdp", "pass_", "dtype"], mask)
        r.to_csv("results/e1_ci.csv", index=False)
        pd.set_option("display.width", 200)
        print(r.round(2).to_string(index=False))
