"""E5（MI350X GP likelihood）摘要：gp_tdp.sh 的 RESULT_GP 每行 = 一張 GPU 上一個變體的六個 range 參數。

usage: gp_summary.py <log id> [--out results/]      例如 gp_1029_0731
配對：同上限、同輪（round）、同 GPU 的兩次 native 平均為分母（一個行程內的順序是 native t36n t28n t28n t36n native，
本身就是 ABBA）。速度 = native wall ÷ 變體 wall；能量用硬體累計器；max|Δℓ| 是相對該行程第一次 native 的最大偏差。
95% t 區間的樣本 = GPU × round（每個上限 16 個）。
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import BDF, ci  # noqa: E402
from smi import load_smi, window_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    smi = load_smi(f"{a.logdir}/{a.id}_smi.csv")
    dev2bdf, rows = {}, []
    for line in open(f"{a.logdir}/{a.id}.out"):
        if line.startswith("n=") and "bdf=" in line:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            dev2bdf[int(kv["gpu"])] = kv["bdf"]
        if not line.startswith("RESULT_GP"):
            continue
        kv = dict(re.findall(r"(\w+)=([^\s,\[]+)", line.replace("tag=", "")))
        dll = [float(x) for x in re.search(r"dll=\[([^\]]*)\]", line).group(1).split()]
        ll = tuple(float(x) for x in re.search(r"loglik=\[([^\]]*)\]", line).group(1).split())
        g = BDF.index(dev2bdf[int(kv["gpu"])])
        w = window_stats(smi, g, float(kv["t_begin"]), float(kv["t_end"]))
        v = kv["variant"] if kv.get("emu", "none") == "none" else kv["emu"]  # GEMMul8 行程記為 g8m<k>
        rows.append(dict(cap=int(kv["cap"]), round=int(kv["round"]), gpu=g, variant=v, pass_=kv["pass"],
                         status=kv["status"], wall=float(kv["wall"].rstrip("s")), dll_max=np.nanmax(np.abs(dll)),
                         ll=ll, **w))
    d = pd.DataFrame(rows)
    # GEMMul8 行程的第一次（pass A）含每個新形狀的冷啟動成本，只取第二次；raw 檔保留 cold 欄
    d["cold"] = d.variant.str.startswith("g8") & (d.pass_ == "A")
    # 行程內的 dll 以該行程第一次 native 為參考；GEMMul8 行程裡的 native 本身已被 emulation，所以改以同一 GPU 的
    # 原生行程（資料由同一 seed 產生，完全相同）為參考，取六個 range 的最大 |Δℓ|
    ref = d[d.variant == "native"].groupby("gpu").ll.first()
    # （loglik 的列印解析度：舊 log 為 1e-6、2026-10-29 21:50 之後 1e-10；非 GEMMul8 的列保留行程內精確的 dll）
    emu = d.variant.str.startswith("g8")
    d.loc[emu, "dll_max"] = [float(np.nanmax(np.abs(np.subtract(r.ll, ref[r.gpu])))) for r in d[emu].itertuples()]
    d_all, d = d, d[~d.cold]
    key = ["cap", "round", "gpu"]
    nat = d[d.variant == "native"].groupby(key)[["wall", "energy_j"]].mean()
    out = []
    for (cap, v), g in d.groupby(["cap", "variant"]):
        gg = g.groupby(key)[["wall", "energy_j", "power_acc_w", "gfxclk", "ppt_frac"]].mean()
        b = nat.reindex(gg.index)
        sp = b.wall.values / gg.wall.values - 1
        en = gg.energy_j.values / b.energy_j.values - 1
        lo, hi = ci(sp)
        out.append(dict(cap=cap, variant=v, n=len(gg), notpd=int((g.status != "ok").sum()), wall=g.wall.mean(),
                        dll_max=g.dll_max.max(), power_w=gg.power_acc_w.mean(), gfxclk=gg.gfxclk.mean(),
                        ppt_frac=gg.ppt_frac.mean(), speedup_pct=100 * sp.mean(), speedup_lo=100 * lo,
                        speedup_hi=100 * hi, energy_chg_pct=100 * en.mean(),
                        cap_ok=bool((g.cap_min == cap).all() and (g.cap_max == cap).all())))
    s = pd.DataFrame(out).sort_values(["cap", "variant"], ascending=[False, True])
    os.makedirs(a.out, exist_ok=True)
    d_all.to_csv(f"{a.out}/{a.id}_raw.csv", index=False)
    s.to_csv(f"{a.out}/{a.id}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
