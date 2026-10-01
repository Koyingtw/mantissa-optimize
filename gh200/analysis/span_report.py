"""Summarise MANT_SPAN files (shim per-row exponent span): for each GEMM shape and operand, the fraction of
nonzero elements more than 24/32/40/48/56 binades below their row's (op(A)) or column's (op(B)) maximum,
i.e. what per-row scaled INT8 emulation with that many retained bits would lose.

usage: span_report.py name=file[,file...] [...]  -> table on stdout; SPAN_OUT=<csv> writes the per-shape rows
"""
import os
import sys

import pandas as pd

COLS = ["op", "m", "n", "k", "ta", "tb", "per", "calls", "gflop", "lines", "nonzero",
        "below24", "below32", "below40", "below48", "below56"]


def load(name, files):
    rows = []
    for f in files:
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            rows.append(dict(zip(COLS, [v[0]] + [float(x) for x in v[1:]]), app=name))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d = pd.concat([load(*a.split("=", 1)[:1], a.split("=", 1)[1].split(",")) for a in sys.argv[1:]], ignore_index=True)
    d = d[d.nonzero > 0]
    for t in (24, 32, 40, 48, 56):
        d[f"f{t}"] = d[f"below{t}"] / d.nonzero
    if os.environ.get("SPAN_OUT"):
        d.to_csv(os.environ["SPAN_OUT"], index=False)
    # flop-weighted over shapes: how much of the GEMM work sees operands with such elements
    out = []
    for (app, op), g in d.groupby(["app", "op"]):
        w = g.gflop / g.gflop.sum()
        r = dict(app=app, op=op, shapes=len(g), gflop=g.gflop.sum())
        for t in (24, 32, 40, 48):
            r[f"flop加權 >{t}"] = (w * g[f"f{t}"]).sum()
            r[f"最差形狀 >{t}"] = g[f"f{t}"].max()
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.3g}"))
    print("\n最嚴重的形狀（>32 比例前 8 名）：")
    top = d.sort_values("f32", ascending=False).head(8)
    print(top[["app", "op", "m", "n", "k", "ta", "tb", "calls", "gflop", "f24", "f32", "f40", "f48"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2e}" if abs(x) < 0.01 else f"{x:.4g}"))
