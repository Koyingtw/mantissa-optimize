"""功耗模型（power_model.py）的獨立評估：P0 不再以驗證資料擬合。

(a) idle：P0 = 獨立量測的閒置功耗（GH200 模組 174 W：E1 job 1667–1669 冷卻期 SM < 500 MHz 的模組功耗中位數；
    MI350X 每卡 262 W：n02 E1 細網格 log 中 gfxclk < 300 MHz、活動 < 5% 的取樣中位數），不擬合任何參數。
(b) loo：leave-one-format-out。對每個被留下的格式（dtype × 保留位元），P0 只用其餘格式的誤差擬合，再預測該格式。
(c) zero：P0 = 0（全部功耗都當成動態）。
每種都回報平均絕對誤差（MAE）、最大絕對誤差與點數。GH200 用 onecap（以最低上限一點校正 R，該點不計），
MI350X 用 fixed+stretch（鎖 1300 MHz 量的 R 與 σ，完全不用掃描資料）。
usage: model_eval.py      → results/model_eval.csv、results/model_eval_points.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from power_model import (RES, gh200_curves, mi350_curves, mi350_fixed, predict, predict_stretch,  # noqa: E402
                         rho_onecap)

P0_IDLE = {"GH200": 174.0, "MI350X": 262.0}
GRID = list(range(0, 351, 10))


def run(plat, p0):
    if plat == "GH200":
        d = predict(gh200_curves(), rho_onecap, p0)
        return d[d.cap != d.groupby(["dtype", "frac"]).cap.transform("min")]  # 校正點不計
    return predict_stretch(mi350_curves(), mi350_fixed(), p0)


def main():
    rows, pts = [], []
    for plat in ("GH200", "MI350X"):
        for name, p0 in (("idle", P0_IDLE[plat]), ("zero", 0.0)):
            d = run(plat, p0)
            rows.append(dict(platform=plat, method=name, p0=p0, mae=d.err_pts.abs().mean(),
                             max=d.err_pts.abs().max(), n=len(d)))
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
    pd.concat(pts).to_csv(f"{RES}/model_eval_points.csv", index=False)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    p = pd.concat(pts)
    for plat, meth in (("GH200", "idle"), ("MI350X", "idle")):
        g = p[(p.platform == plat) & (p.method == meth)]
        print(f"\n{plat} {meth}: |err| by format (mean / max)")
        print(g.groupby(["dtype", "frac"]).err_pts.agg(lambda x: f"{x.abs().mean():.1f} / {x.abs().max():.1f}").to_string())


if __name__ == "__main__":
    main()
