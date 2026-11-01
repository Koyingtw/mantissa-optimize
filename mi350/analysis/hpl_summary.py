"""E2/E3（MI350X HPL-MxP／FP64 HPL）摘要：解析 hpl_tdp.sh 的 === RUN / === END 區塊，對齊功耗 log。

usage: hpl_summary.py <log id> [...] [--out results/]      例如 mxp_1028_2350
每次執行一列（results/<id>_raw.csv）：成績（HPL-MxP 用 reportable GFLOPS，含迭代修正；另記 LU-only）、
scaled residual、PASSED、8 卡合計功耗與能量（硬體累計器）、平均 gfxclk、PPT 駐留比例。
摘要（results/<id>_summary.csv）：cap × variant，相對同一 cap、同一 ABBA 區塊（A<r>/B<r> 配成一組）的原生
平均之比值，95% t 區間（樣本 = 每個 pass），以及 ΔW。只有 PASSED 的執行計入；未通過的列在 failed 欄。
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import ci, delta_w  # noqa: E402
from smi import load_smi, node_window_stats  # noqa: E402

NATIVE = {"mxp": "native", "hpl": "53:53"}


def parse(path):
    runs, cur = [], None
    for line in open(path, errors="replace"):
        m = re.match(r"=== (RUN|END) (.*)", line)
        if m:
            kv = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
            if m.group(1) == "RUN":
                cur = dict(bench=kv["bench"], cap=int(kv["cap"]), blk=int(kv.get("blk", 0)), pass_=kv["pass"],
                           variant=kv["variant"],
                           t_begin=float(kv["t"]), gflops=np.nan, lu_gflops=np.nan, resid=np.nan, passed=False)
            elif cur is not None:
                cur.update(t_end=float(kv["t"]), rc=int(kv["rc"]))
                runs.append(cur)
                cur = None
            continue
        if cur is None:
            continue
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        if m := re.search(r"^\s*GFLOPS = ([0-9.e+]+)", line):          # HPL-MxP reportable
            cur["gflops"] = float(m.group(1))
        elif m := re.search(r"LU GFLOPS = ([0-9.e+]+)", line):
            cur["lu_gflops"] = float(m.group(1))
        elif m := re.match(r"^WC\S+\s+\d+\s+\d+\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.e+]+)", line):  # rocHPL
            cur["gflops"] = float(m.group(2))
            cur["hpl_time"] = float(m.group(1))
        elif "PASSED" in line or "FAILED" in line:
            if m := re.search(r"=\s*([0-9.eE+-]+)\s*\.+\s*(PASSED|FAILED)", line):
                cur["resid"] = float(m.group(1))
                cur["passed"] = m.group(2) == "PASSED"
    return pd.DataFrame(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = []
    for i in a.ids:
        d = parse(f"{a.logdir}/{i}.out")
        smi = load_smi(f"{a.logdir}/{i}_smi.csv")
        gpus = sorted(smi.gpu.unique())  # 8, or 16 for two-node runs (scripts/hpl2_tdp.sh)
        st = [node_window_stats(smi, r.t_begin, r.t_end, gpus) for r in d.itertuples()]
        d = pd.concat([d.reset_index(drop=True), pd.DataFrame(st)], axis=1)
        d["id"] = i
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["tflops"] = d.gflops / 1e3
    # 配對區塊：同一個 log、同一個 PAIRWISE 區塊（blk）、同一輪 ABBA（A0/B0 → 0）
    d["block"] = d.id + ":" + d.blk.astype(str) + ":" + d.pass_.str[1:]
    bench = d.bench.iloc[0]
    nat = NATIVE[bench]
    ok = d[d.passed & (d.pass_ != "W")]
    out = []
    for (cap, v), g in ok.groupby(["cap", "variant"]):
        base = ok[(ok.cap == cap) & (ok.variant == nat)]
        r = [x.tflops / base[base.block == x.block].tflops.mean() - 1 for x in g.itertuples()]
        e = [x.energy_j / base[base.block == x.block].energy_j.mean() - 1 for x in g.itertuples()]
        lo, hi = ci(r)
        out.append(dict(cap=cap, variant=v, n=len(g), failed=int(((d.cap == cap) & (d.variant == v) & ~d.passed &
                                                                  (d.pass_ != "W")).sum()),
                        tflops=g.tflops.mean(), lu_tflops=g.lu_gflops.mean() / 1e3, resid_max=g.resid.max(),
                        power_w=g.power_acc_w.mean(), gfxclk=g.gfxclk.mean(), ppt_frac=g.ppt_frac.mean(),
                        t_hot_max=g.t_hot_max.max(), speedup_pct=100 * np.nanmean(r), speedup_lo=100 * lo,
                        speedup_hi=100 * hi, energy_chg_pct=100 * np.nanmean(e),
                        cap_ok=bool((g.cap_min == cap).all() and (g.cap_max == cap).all())))
    s = pd.DataFrame(out)
    curve = [tuple(x) for x in s[s.variant == nat][["cap", "tflops"]].values]
    dws = []
    for row in s.itertuples():
        dw, lower = delta_w(curve, row.tflops, row.cap)
        dws.append((">" if lower else "") + ("" if np.isnan(dw) else f"{dw:.0f}"))
    s["delta_w_per_gpu"] = dws
    s = s.sort_values(["cap", "variant"], ascending=[False, True])
    tag = a.ids[0]
    d.to_csv(f"{a.out}/{tag}_raw.csv", index=False)
    s.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
