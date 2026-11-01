"""E1（MI350X GEMM）摘要：每個 RESULT 對齊功耗 log，再算相對原生位元數的配對比值。

usage: e1_summary.py <log id> [<log id> ...] [--out results/]     例如 e1_1028_1709
輸出：results/<id>_raw.csv（每 GPU 每次量測一列）、results/<id>_summary.csv（cap × dtype × frac × low）。
配對：同一 cap、同一 pass、同一 GPU 的原生設定（frac = 原生、low = zero）為分母；
95% 信賴區間用 t 分佈，樣本 = GPU × pass（GPU 間獨立性的假設見實驗日誌）。
ΔW：在原生的「cap → TFLOP/s」曲線上內插出原生要多少上限才能追上縮減後的速度，再減去目前上限。
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi, window_stats  # noqa: E402

NATIVE = {"fp64": 52, "fp32": 23, "bf16": 7, "fp8": 3, "dfma": 52, "nbody": 52, "copy": 52, "triad": 52}
BDF = ["0000:09:00.0", "0000:19:00.0", "0000:69:00.0", "0000:79:00.0",
       "0000:89:00.0", "0000:99:00.0", "0000:e9:00.0", "0000:f9:00.0"]  # 取樣器的 gpu 欄 = 此清單索引


def load(ids, logdir):
    rows = []
    for i in ids:
        smi = load_smi(f"{logdir}/{i}_smi.csv")
        for line in open(f"{logdir}/{i}.out"):
            if not line.startswith("RESULT"):
                continue
            kv = dict(re.findall(r"(\w+)=([^\s,]+)", line.replace("tag=", "")))
            g = BDF.index(kv["bdf"])  # amd-smi／取樣器序；bench 的 --device 序依 HIP_VISIBLE_DEVICES 而定
            w = window_stats(smi, g, float(kv["t_begin"]), float(kv["t_end"]))
            dtype, frac = kv["dtype"], int(kv["frac"])
            emu = kv.get("emu", "none")
            if emu.startswith("g8m"):  # GEMMul8 emulation：dtype g8、frac 欄記 moduli 數；輸入有截斷時記成 g8t<輸入位元>
                dtype, frac = ("g8" if frac == 52 else f"g8t{frac}"), int(emu[3:])
            rows.append(dict(id=i, cap=int(kv["cap"]), pass_=kv["pass"], gpu=g, dtype=dtype, frac=frac,
                             low=kv["low"], m=int(kv["m"]), n=int(kv["n"]), k=int(kv["k"]),
                             tflops=float(kv["tflops"]), rel_err=float(kv.get("rel_err", "nan")),
                             spread=float(kv.get("spread", 0)), smode=kv.get("smode", "elem"),
                             err_p50=float(kv.get("err_p50", "nan")), err_p99=float(kv.get("err_p99", "nan")),
                             tog=(float(kv.get("tog_a", -1)) + float(kv.get("tog_b", -1))) / 2,
                             hw=(float(kv.get("hw_a", -1)) + float(kv.get("hw_b", -1))) / 2,
                             t_begin=float(kv["t_begin"]), t_end=float(kv["t_end"]), **w))
    d = pd.DataFrame(rows)
    d["gflops_per_j"] = d.tflops * 1e3 / d.power_acc_w
    d["base"] = d.dtype.where(~d.dtype.str.startswith("g8"), "fp64")  # 配對用的原生格式
    return d


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def delta_w(native_curve, perf, cap):
    """native_curve: [(cap, tflops)]，依 cap 遞增；回傳 (ΔW, 是否只是下限)。"""
    c = sorted(native_curve)
    caps = np.array([a for a, _ in c], float)
    perfs = np.array([b for _, b in c], float)
    if perf > perfs.max():
        return caps.max() - cap, True
    if perf < perfs.min():
        return np.nan, False
    return float(np.interp(perf, perfs, caps)) - cap, False


def summarize(d):
    key = ["cap", "pass_", "gpu", "base", "m", "n", "k", "spread", "smode"]
    nat = d[(d.dtype == d.base) & (d.frac == d.base.map(NATIVE)) & (d.low == "zero")].set_index(key)
    out = []
    for (cap, dtype, frac, low, m, n, k, spread, smode), g in d.groupby(["cap", "dtype", "frac", "low", "m", "n", "k",
                                                                         "spread", "smode"]):
        b = nat.reindex(g.set_index(key).index)
        r_perf = g.tflops.values / b.tflops.values
        r_pow = g.power_acc_w.values / b.power_acc_w.values
        r_eff = g.gflops_per_j.values / b.gflops_per_j.values
        lo, hi = ci(r_perf - 1)
        out.append(dict(cap=cap, dtype=dtype, frac=frac, low=low, m=m, n=n, k=k, spread=spread, smode=smode,
                        tog=g.tog.mean(), hw=g.hw.mean(), rel_err=g.rel_err.mean(), err_p50=g.err_p50.median(), err_p99=g.err_p99.max(),
                        n_meas=len(g), tflops=g.tflops.mean(),
                        power_w=g.power_acc_w.mean(), gfxclk=g.gfxclk.mean(), ppt_frac=g.ppt_frac.mean(),
                        thm_frac=g.thm_frac.mean(), t_hot_max=g.t_hot_max.max(), gflops_per_j=g.gflops_per_j.mean(),
                        speedup_pct=100 * (np.nanmean(r_perf) - 1), speedup_lo=100 * lo, speedup_hi=100 * hi,
                        power_chg_pct=100 * (np.nanmean(r_pow) - 1), eff_chg_pct=100 * (np.nanmean(r_eff) - 1),
                        cap_ok=bool((g.cap_min == cap).all() and (g.cap_max == cap).all())))
    s = pd.DataFrame(out)
    # ΔW：每個 dtype 的原生曲線
    dws = []
    for _, r in s.iterrows():
        bt = "fp64" if r["dtype"].startswith("g8") else r["dtype"]
        nc = s[(s.dtype == bt) & (s.frac == NATIVE[bt]) & (s.low == "zero") & (s.k == r["k"]) & (s.m == r["m"]) &
               (s.spread == r["spread"]) & (s.smode == r["smode"])]
        dw, lower = delta_w([tuple(x) for x in nc[["cap", "tflops"]].values], r["tflops"], r["cap"])
        dws.append((">" if lower else "") + ("" if np.isnan(dw) else f"{dw:.0f}"))
    s["delta_w"] = dws
    return s.sort_values(["dtype", "spread", "smode", "k", "cap", "low", "frac"],
                         ascending=[True, True, True, True, False, True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.ids, a.logdir)
    s = summarize(d)
    tag = a.ids[0]
    d.to_csv(f"{a.out}/{tag}_raw.csv", index=False)
    s.to_csv(f"{a.out}/{tag}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
