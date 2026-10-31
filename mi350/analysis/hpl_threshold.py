"""HPL on MI350X: each method's fastest measured configuration that meets a common residual threshold (review round 3).

Every HPL run on MI350X (bit sweeps at three N, the 1x8 cap sweeps on both nodes, the 4x2 configurations used for
GEMMul8) is a candidate. For a threshold tau on HPL's scaled residual (16 = HPL's own check) and a cap, each method
(native, truncation at any bit count, GEMMul8 at any number of moduli) takes its fastest configuration whose residual is
below tau; its rate is divided by the fastest native configuration measured on the same node at the same cap, so node
differences (about 1%) cancel. Residuals are deterministic per configuration (repeated runs reproduce them exactly).
usage: hpl_threshold.py → results/hpl_threshold.csv
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

RUNS = [  # result id, node, configuration
    ("hplbits262k_1029_0827", "n01", "N=262144, 1x8"),
    ("hplbits393k_1029_0854", "n01", "N=393216, 1x8"),
    ("hplbits540k_1029_0924", "n01", "N=540672, 1x8"),
    ("hpl_1029_0336", "n01", "N=540672, 1x8"),
    ("hpl3A_1029_1914", "n01", "N=540672, 1x8"),
    ("hpl3A2_1030_0128", "n01", "N=458752, 1x8"),
    ("hpl3B2_1030_0634", "n01", "N=458752, 4x2"),
    ("hpln02_1029_1045", "n02", "N=540672, 1x8"),
    ("hpl3B_1029_1914", "n02", "N=491520, 4x2"),
]
TAUS = [16, 1, 0.1]


def method(v):
    return "native" if v == "53:53" else "emulation" if v.startswith("g8") else "truncation"


def main():
    rows = []
    for rid, node, cfg in RUNS:
        d = pd.read_csv(f"{RES}/{rid}_raw.csv")
        d = d[d.passed & (d.pass_ != "W")]
        for (cap, v), g in d.groupby(["cap", "variant"]):
            rows.append(dict(node=node, cfg=cfg, rid=rid, cap=cap, variant=v, method=method(v), runs=len(g),
                             tflops=g.gflops.mean() / 1000, resid=g.resid.max()))
    c = pd.DataFrame(rows)
    best_native = c[c.method == "native"].groupby(["node", "cap"]).tflops.max()
    c["gain_pct"] = 100 * (c.tflops / best_native.reindex(pd.MultiIndex.from_frame(c[["node", "cap"]])).values - 1)
    out = []
    for tau in TAUS:
        for cap in sorted(c.cap.unique(), reverse=True):
            for m in ("truncation", "emulation"):
                e = c[(c.cap == cap) & (c.method == m) & (c.resid < tau)]
                if e.empty:
                    out.append(dict(tau=tau, cap=cap, method=m))
                    continue
                b = e.loc[e.gain_pct.idxmax()]
                out.append(dict(tau=tau, cap=cap, method=m, variant=b.variant, node=b.node, cfg=b.cfg, resid=b.resid,
                                tflops=b.tflops, gain_pct=b.gain_pct, rid=b.rid))
    o = pd.DataFrame(out)
    o.to_csv(f"{RES}/hpl_threshold.csv", index=False)
    c.sort_values(["cap", "gain_pct"], ascending=[False, False]).to_csv(f"{RES}/hpl_threshold_candidates.csv", index=False)
    pd.set_option("display.width", 200)
    print(o.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nfastest native per node and cap (TFLOP/s):")
    print(best_native.unstack(0).to_string())


if __name__ == "__main__":
    main()
