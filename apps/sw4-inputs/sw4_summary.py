#!/usr/bin/env python3
"""SW4 執行摘要：GPU 功耗/時脈/使用率（nvidia-smi CSV）與接收站時間序列的正確性量。
用法: sw4_summary.py <run_dir> [t0 t1]
"""
import csv, glob, hashlib, math, os, sys

run = sys.argv[1]
os.chdir(run)

# --- GPU log ---
try:
    rows = list(csv.reader(open("gpu.csv")))[1:]
    def num(s):
        return float(s.strip().split()[0])
    P = [num(r[1]) for r in rows if "W" in r[1]]
    C = [num(r[2]) for r in rows]
    U = [num(r[3]) for r in rows]
    busy = [i for i, u in enumerate(U) if u >= 50]
    msg = f"GPU samples={len(P)} meanPower={sum(P)/len(P):.1f}W maxPower={max(P):.1f}W meanUtil={sum(U)/len(U):.1f}% meanSM={sum(C)/len(C):.0f}MHz"
    if busy:
        Pb = [P[i] for i in busy]; Cb = [C[i] for i in busy]; Ub = [U[i] for i in busy]
        msg += f" | busy(util>=50%) samples={len(busy)} meanPower={sum(Pb)/len(Pb):.1f}W meanUtil={sum(Ub)/len(Ub):.1f}% meanSM={sum(Cb)/len(Cb):.0f}MHz minSM={min(Cb):.0f}MHz"
    print(msg)
except Exception as e:
    print("gpu.csv parse failed:", e)

if len(sys.argv) > 3:
    print(f"wall(mpirun)={float(sys.argv[3]) - float(sys.argv[2]):.2f}s")

# --- receiver time series (usgs text format: t ux uy uz ...) ---
files = sorted(glob.glob("out-*/*.txt"))
h = hashlib.md5()
for f in files:
    data = []
    for line in open(f):
        parts = line.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if len(vals) >= 4:
            data.append(vals)
    # 只雜湊資料列（檔頭 "# Date: ..." 會隨執行時間改變）
    h.update(b"".join(l for l in open(f, "rb") if not l.startswith(b"#")))
    if not data:
        print(f"{os.path.basename(f)}: no data")
        continue
    ncomp = len(data[0]) - 1
    out = [f"{os.path.basename(f)} n={len(data)}"]
    for c in range(1, ncomp + 1):
        col = [r[c] for r in data]
        l2 = math.sqrt(sum(x * x for x in col))
        mx = max(abs(x) for x in col)
        out.append(f"c{c}: max|u|={mx:.17g} l2={l2:.17g}")
    print("  ".join(out))
print(f"receiver files={len(files)} md5(data lines)={h.hexdigest()}")
