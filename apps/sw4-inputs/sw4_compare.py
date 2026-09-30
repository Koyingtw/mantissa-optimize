#!/usr/bin/env python3
"""比較兩個 SW4 run 的接收站時間序列：每站 max_t |u_b - u_a| / max_t |u_a|（三分量取最大）。
用法: sw4_compare.py <run_a> <run_b>"""
import glob, os, sys
def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "out-*", "*.txt"))):
        rows = []
        for l in open(f):
            if l.startswith("#"): continue
            try: rows.append([float(x) for x in l.split()])
            except ValueError: pass
        out[os.path.basename(f)] = rows
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
worst = 0.0
for k in a:
    ra, rb = a[k], b.get(k)
    if rb is None or len(rb) != len(ra):
        print(k, "missing/length mismatch"); continue
    rel = []
    for c in range(1, len(ra[0])):
        amax = max(abs(r[c]) for r in ra)
        dmax = max(abs(x[c] - y[c]) for x, y in zip(ra, rb))
        rel.append(dmax / amax if amax else 0.0)
    worst = max(worst, max(rel))
    print(f"{k}: rel.maxdiff per comp = " + " ".join(f"{r:.3e}" for r in rel))
print(f"worst rel.maxdiff = {worst:.3e}  (0 表示逐位相同)")
