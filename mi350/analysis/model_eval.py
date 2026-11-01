"""功耗模型（power_model.py）的獨立評估：P0 不再以驗證資料擬合。

(a) idle：P0 = 獨立量測的閒置功耗（GH200 模組 174 W：E1 job 1667–1669 冷卻期 SM < 500 MHz 的模組功耗中位數；
    MI350X 每卡 262 W：n02 E1 細網格 log 中 gfxclk < 300 MHz、活動 < 5% 的取樣中位數），不擬合任何參數。
(b) loo：leave-one-format-out。對每個被留下的格式（dtype × 保留位元），P0 只用其餘格式的誤差擬合，再預測該格式。
(c) zero：P0 = 0（全部功耗都當成動態）。
(d) 對照的簡單預測法（審稿第五輪 M3），與模型在同一組點上比較：
    const ：在一個上限量到的加速套用到所有上限（GH200 用最低上限，與模型的校正點相同；MI350X 用預設的 1000 W，
            使用者最容易量到），該點不計；
    linear：以最低與最高兩個上限量到的加速做線性內插／外插，兩點不計；
    energy：只用鎖頻量測（MI350X）：加速 = 原生每次運算的能量 ÷ 截斷版的能量 − 1（功耗與時脈成正比、沒有 P0）；
    clock ：只用鎖頻量測（MI350X）：加速 = σ − 1。
每種都回報平均絕對誤差（MAE）、最大絕對誤差與點數。GH200 用 onecap（以最低上限一點校正 R，該點不計），
MI350X 用 fixed+stretch（鎖 1300 MHz 量的 R 與 σ，完全不用掃描資料）。
usage: model_eval.py      → results/model_eval.csv、results/model_eval_points.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from power_model import (NATIVE, RES, gh200_curves, mi350_curves, mi350_fixed, native_perf,  # noqa: E402
                         predict, predict_stretch, rho_onecap)

P0_IDLE = {"GH200": 174.0, "MI350X": 262.0}
GRID = list(range(0, 351, 10))


def run(plat, p0):
    if plat == "GH200":
        d = predict(gh200_curves(), rho_onecap, p0)
        return d[d.cap != d.groupby(["dtype", "frac"]).cap.transform("min")]  # 校正點不計
    return predict_stretch(mi350_curves(), mi350_fixed(), p0)


def measured(plat):
    cur = gh200_curves() if plat == "GH200" else mi350_curves()
    out = []
    for (dt, fr), g in cur.groupby(["dtype", "frac"]):
        if fr == NATIVE[dt]:
            continue
        nc = cur[(cur.dtype == dt) & (cur.frac == NATIVE[dt])].sort_values("cap")
        for r in g.itertuples():
            out.append(dict(dtype=dt, frac=fr, cap=r.cap, meas_gain=100 * (r.perf / native_perf(nc, r.cap) - 1)))
    return pd.DataFrame(out).dropna()


def baselines(plat, model):
    """Simple predictors, each scored on its own held-out points together with the model on the same points."""
    m = measured(plat)
    key = ["dtype", "frac", "cap"]
    mod = model.set_index(key).pred_gain
    rows = []

    def score(name, pred, keep):
        e = pred[keep].dropna()
        idx = pd.MultiIndex.from_frame(e[key])
        em = (mod.reindex(idx).values - e.meas_gain.values)
        rows.append(dict(platform=plat, baseline=name, n=len(e), mae=(e.pred_gain - e.meas_gain).abs().mean(),
                         max=(e.pred_gain - e.meas_gain).abs().max(), model_mae=np.nanmean(np.abs(em)),
                         model_max=np.nanmax(np.abs(em))))
    ref_cap = m.groupby(["dtype", "frac"]).cap.transform("min" if plat == "GH200" else "max")
    ref = m[m.cap == ref_cap].set_index(["dtype", "frac"]).meas_gain
    c = m.assign(pred_gain=ref.reindex(pd.MultiIndex.from_frame(m[["dtype", "frac"]])).values)
    score("const", c, c.cap != ref_cap)
    lo = m.groupby(["dtype", "frac"]).cap.transform("min")
    hi = m.groupby(["dtype", "frac"]).cap.transform("max")
    lin = []
    for (dt, fr), g in m.groupby(["dtype", "frac"]):
        a, b = g[g.cap == g.cap.min()].iloc[0], g[g.cap == g.cap.max()].iloc[0]
        lin.append(g.assign(pred_gain=a.meas_gain + (b.meas_gain - a.meas_gain) * (g.cap - a.cap) / (b.cap - a.cap)))
    lin = pd.concat(lin)
    score("linear", lin, (lin.cap != lo.loc[lin.index]) & (lin.cap != hi.loc[lin.index]))
    if plat == "MI350X":
        fx = mi350_fixed().set_index(["dtype", "frac"])
        e, k = [], []
        for r in m.itertuples():
            a, b = fx.loc[(r.dtype, r.frac)], fx.loc[(r.dtype, NATIVE[r.dtype])]
            e.append(100 * ((b.power_w / b.tflops) / (a.power_w / a.tflops) - 1))
            k.append(100 * (a.gfxclk / b.gfxclk - 1))
        score("energy", m.assign(pred_gain=e), m.cap > 0)
        score("clock", m.assign(pred_gain=k), m.cap > 0)
    return rows


def main():
    rows, pts, base = [], [], []
    for plat in ("GH200", "MI350X"):
        for name, p0 in (("idle", P0_IDLE[plat]), ("zero", 0.0)):
            d = run(plat, p0)
            rows.append(dict(platform=plat, method=name, p0=p0, mae=d.err_pts.abs().mean(),
                             max=d.err_pts.abs().max(), n=len(d)))
            if name == "idle":
                base += baselines(plat, d)
            pts.append(d.assign(platform=plat, method=name, p0=p0))
        # leave-one-format-out：每個格式的 P0 由其他格式擬合
        cache = {p0: run(plat, p0) for p0 in GRID if not (plat == "GH200" and p0 >= 300)}
        fmts = sorted({(r.dtype, r.frac) for r in cache[0].itertuples()})
        held = []
        for f in fmts:
            best = min(cache, key=lambda p0: cache[p0][(cache[p0].dtype != f[0]) | (cache[p0].frac != f[1])]
                       .err_pts.abs().mean())
            g = cache[best]
            held.append(g[(g.dtype == f[0]) & (g.frac == f[1])].assign(p0=best))
        h = pd.concat(held)
        rows.append(dict(platform=plat, method="loo", p0=f"{h.p0.min():.0f}-{h.p0.max():.0f}",
                         mae=h.err_pts.abs().mean(), max=h.err_pts.abs().max(), n=len(h)))
        pts.append(h.assign(platform=plat, method="loo"))
    r = pd.DataFrame(rows)
    r.to_csv(f"{RES}/model_eval.csv", index=False)
    b = pd.DataFrame(base)
    b.to_csv(f"{RES}/model_baselines.csv", index=False)
    print(b.to_string(index=False, float_format=lambda x: f"{x:.2f}"), "\n")
    pd.concat(pts).to_csv(f"{RES}/model_eval_points.csv", index=False)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p = pd.concat(pts)
    for plat, meth in (("GH200", "idle"), ("MI350X", "idle")):
        g = p[(p.platform == plat) & (p.method == meth)]
        print(f"\n{plat} {meth}: |err| by format (mean / max)")
        print(g.groupby(["dtype", "frac"]).err_pts.agg(lambda x: f"{x.abs().mean():.1f} / {x.abs().max():.1f}").to_string())


if __name__ == "__main__":
    main()
