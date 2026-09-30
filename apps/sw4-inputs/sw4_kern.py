#!/usr/bin/env python3
"""nsys cuda_gpu_kern_sum CSV → 依「lambda 所在的 SW4 函式」彙總 kernel 時間。
用法: sw4_kern.py kern_cuda_gpu_kern_sum.csv
"""
import csv, re, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
agg = defaultdict(lambda: [0.0, 0])
for r in rows:
    n = r["Name"]
    # RAJA/forall3 lambda 會帶出所在函式，例如 rhs4th3fortsgstr_ci(...)::{lambda
    m = re.findall(r"([A-Za-z_][A-Za-z0-9_:]*)\([^()]*\)::\{?lambda", n)
    key = m[0] if m else re.sub(r"<.*", "", n)[:60]
    agg[key][0] += float(r["Total Time (ns)"])
    agg[key][1] += int(r["Instances"])
print(f"GPU kernel total = {tot/1e9:.2f} s")
for k, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:20]:
    print(f"  {t/1e9:8.3f}s {100*t/tot:5.1f}% n={c:>7}  {k}")
