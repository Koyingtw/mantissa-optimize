"""E6（MI350X DF-MP2）摘要：mp2_tdp.sh 的 RESULT_MP2 每行 = 一張 GPU 上一個變體的完整 MP2 收縮。

usage: mp2_summary.py <log id> [--out results/]      例如 mp2_1029_1700
配對：同上限、同輪、同 GPU 的兩次 native（emu=none 行程）平均為分母；GEMMul8 的行程（emu=g8m<k>）記為變體 g8m<k>。
速度 = native wall ÷ 變體 wall；|ΔE| = 相對同 GPU 原生相關能的最大偏差（E_h）；能量用硬體累計器。
95% t 區間的樣本 = GPU × 輪。
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
        if line.startswith("nocc=") and "bdf=" in line:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            dev2bdf[int(kv["gpu"])] = kv["bdf"]
        if not line.startswith("RESULT_MP2"):
            continue
        kv = dict(re.findall(r"(\w+)=([^\s,]+)", line.replace("tag=", "")))
        g = BDF.index(dev2bdf[int(kv["gpu"])])
        v = kv["variant"] if kv["emu"] == "none" else kv["emu"]
        w = window_stats(smi, g, float(kv["t_begin"]), float(kv["t_end"]))
        rows.append(dict(cap=int(kv["cap"]), round=int(kv["round"]), gpu=g, variant=v, pass_=kv["pass"],
                         e_corr=float(kv["e_corr"]), wall=float(kv["wall"].rstrip("s")), **w))
    d = pd.DataFrame(rows)
    # GEMMul8 行程的第一次（pass A）含每個新形狀的冷啟動成本，只取第二次；冷啟動另記在 raw 檔
    d["cold"] = d.variant.str.startswith("g8") & (d.pass_ == "A")
    d_all, d = d, d[~d.cold]
    key = ["cap", "round", "gpu"]
    nat = d[d.variant == "native"].groupby(key)[["wall", "energy_j", "e_corr"]].mean()
    out = []
    for (cap, v), g in d.groupby(["cap", "variant"]):
        gg = g.groupby(key)[["wall", "energy_j", "power_acc_w", "gfxclk", "ppt_frac", "e_corr"]].mean()
        b = nat.reindex(gg.index)
        sp = b.wall.values / gg.wall.values - 1
        en = gg.energy_j.values / b.energy_j.values - 1
        lo, hi = ci(sp)
        de = np.abs(g.e_corr.values - nat.reindex(g.set_index(key).index).e_corr.values)
        out.append(dict(cap=cap, variant=v, n=len(gg), wall=g.wall.mean(), de_max=de.max(),
                        power_w=gg.power_acc_w.mean(), gfxclk=gg.gfxclk.mean(), ppt_frac=gg.ppt_frac.mean(),
                        speedup_pct=100 * sp.mean(), speedup_lo=100 * lo, speedup_hi=100 * hi,
                        energy_chg_pct=100 * en.mean(),
                        cap_ok=bool((g.cap_min == cap).all() and (g.cap_max == cap).all())))
    s = pd.DataFrame(out).sort_values(["cap", "variant"], ascending=[False, True])
    os.makedirs(a.out, exist_ok=True)
    d_all.to_csv(f"{a.out}/{a.id}_raw.csv", index=False)
    s.to_csv(f"{a.out}/{a.id}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
