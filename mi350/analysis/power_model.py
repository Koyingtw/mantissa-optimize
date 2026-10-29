"""功耗上限下截斷收益的預測模型，在 MI350X（n02 細網格）與 GH200（gh200/results/e1_summary.csv）上驗證。

模型：受限時 P = P0 + c·g(f)，g(f) = V(f)²f 是硬體的 DVFS 曲線（所有設定共用），截斷只改變切換係數 c。
令 ρ = c_reduced / c_native，則截斷後在上限 P 的表現 = 原生在上限 P' = P0 + (P − P0)/ρ 的表現（性能 ∝ f），
所以 ΔW = (P − P0)(1/ρ − 1)；若 P' 超過原生曲線的最大上限，則以該設定的時脈上限性能為頂（clock-bound）。

ρ 的兩種取得方式：
  fixed : 固定時脈的每 FLOP 動態能量比 [(P_r − P0)/perf_r] / [(P_n − P0)/perf_n]（只有 MI350，e1b 資料）
  onecap: 在最低上限的一點，由實測 perf_r 反查原生曲線得 P'_c，ρ = (P_c − P0)/(P'_c − P0)
P0：每個平台一個值。MI350 用 8 卡在 900 MHz 鎖頻、低負載下的功耗不可得，所以掃描 P0 並回報誤差最小值，
同時列出 P0 = 0（純動態）作為對照；P0 只影響 ρ 的換算，不影響「同一 ρ 預測所有上限」這件事。

usage: power_model.py [--p0-mi350 W] [--p0-gh200 W]
輸出 results/model_pred.csv（每個 dtype × frac × cap 的預測與實測）與 results/model_summary.csv（各法誤差）。
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
NATIVE = {"fp64": 52, "bf16": 7, "fp8": 3}


def mi350_curves():
    s = pd.read_csv(f"{RES}/e1n02_1029_0502_summary.csv")
    s = s[s.low == "zero"]
    return s.rename(columns={"tflops": "perf"})[["dtype", "frac", "cap", "perf", "power_w", "gfxclk"]]


def gh200_curves():
    s = pd.read_csv(f"{HERE}/../../gh200/results/e1_summary.csv").drop_duplicates(["dtype", "frac", "tdp"])
    s = s.rename(columns={"tdp": "cap", "mod_w": "power_w", "sm_mhz": "gfxclk"})
    return s[["dtype", "frac", "cap", "perf", "power_w", "gfxclk"]]


def mi350_fixed():
    """固定時脈（鎖 1300 MHz）每個設定的 tflops、power（每 GPU 平均）與實際有效時脈 gfxclk。"""
    files = [f for f in os.listdir(RES) if f.startswith("e1b_clk1300") and f.endswith("_summary.csv")]
    s = pd.concat([pd.read_csv(f"{RES}/{f}") for f in files])
    s = s[s.low == "zero"].groupby(["dtype", "frac"])[["tflops", "power_w", "gfxclk"]].mean().reset_index()
    return s


def predict_stretch(curves, fixed, p0):
    """含時脈伸縮（clock stretching）的版本：鎖在同一個目標頻率時，有效時脈仍依資料而不同（MI350X 實測，
    且不被任何節流計數器記錄）。令 σ = 有效時脈比、R = 動態功耗比（皆在同一鎖定頻率量），則
    perf_r(P) = σ · perf_n(P0 + (P − P0)/R)。σ = 1 時退化為原模型（R = ρ）。"""
    rows = []
    for (dt, fr), g in curves.groupby(["dtype", "frac"]):
        if fr == NATIVE[dt]:
            continue
        a = fixed[(fixed.dtype == dt) & (fixed.frac == fr)]
        b = fixed[(fixed.dtype == dt) & (fixed.frac == NATIVE[dt])]
        if a.empty or b.empty:
            continue
        sigma = a.gfxclk.iloc[0] / b.gfxclk.iloc[0]
        R = (a.power_w.iloc[0] - p0) / (b.power_w.iloc[0] - p0)
        nc = curves[(curves.dtype == dt) & (curves.frac == NATIVE[dt])].sort_values("cap")
        pmax = g.perf.max()
        for r in g.itertuples():
            pn = native_perf(nc, r.cap)
            pp = p0 + (r.cap - p0) / R
            base = native_perf(nc, pp)
            if np.isnan(base):
                c = nc.iloc[-2:]
                slope = (c.perf.iloc[1] - c.perf.iloc[0]) / (c.cap.iloc[1] - c.cap.iloc[0])
                base = c.perf.iloc[1] + slope * (pp - c.cap.iloc[1])
            pred = min(sigma * base, pmax)
            rows.append(dict(dtype=dt, frac=fr, cap=r.cap, rho=R, sigma=sigma, p_equiv=pp,
                             meas_gain=100 * (r.perf / pn - 1), pred_gain=100 * (pred / pn - 1), dw_pred=pp - r.cap))
    d = pd.DataFrame(rows)
    d["err_pts"] = d.pred_gain - d.meas_gain
    return d


def native_perf(nc, cap):
    """原生曲線（cap 遞增）在 cap 的性能；超出範圍回 NaN。"""
    if cap < nc.cap.min() - 1e-9 or cap > nc.cap.max() + 1e-9:
        return np.nan
    return float(np.interp(cap, nc.cap, nc.perf))


def native_cap(nc, perf):
    if perf < nc.perf.min() or perf > nc.perf.max():
        return np.nan
    return float(np.interp(perf, nc.perf, nc.cap))


def predict(curves, rho_of, p0):
    rows = []
    for (dt, fr), g in curves.groupby(["dtype", "frac"]):
        if fr == NATIVE[dt]:
            continue
        nc = curves[(curves.dtype == dt) & (curves.frac == NATIVE[dt])].sort_values("cap")
        rho = rho_of(dt, fr, g, nc, p0)
        if rho is None or not np.isfinite(rho):
            continue
        pmax = g.perf.max()  # 這個設定實測到的最高性能（時脈上限）
        for r in g.itertuples():
            pn = native_perf(nc, r.cap)
            pp = p0 + (r.cap - p0) / rho
            pred = native_perf(nc, pp)
            if np.isnan(pred):  # 超出原生曲線：外插最後一段，並以時脈上限封頂
                c = nc.iloc[-2:]
                slope = (c.perf.iloc[1] - c.perf.iloc[0]) / (c.cap.iloc[1] - c.cap.iloc[0])
                pred = c.perf.iloc[1] + slope * (pp - c.cap.iloc[1])
            pred = min(pred, pmax)
            rows.append(dict(dtype=dt, frac=fr, cap=r.cap, rho=rho, p_equiv=pp, meas_gain=100 * (r.perf / pn - 1),
                             pred_gain=100 * (pred / pn - 1), dw_pred=pp - r.cap))
    d = pd.DataFrame(rows)
    d["err_pts"] = d.pred_gain - d.meas_gain
    return d


def rho_fixed(fixed):
    def f(dt, fr, g, nc, p0):
        a = fixed[(fixed.dtype == dt) & (fixed.frac == fr)]
        b = fixed[(fixed.dtype == dt) & (fixed.frac == NATIVE[dt])]
        if a.empty or b.empty:
            return None
        return ((a.power_w.iloc[0] - p0) / a.tflops.iloc[0]) / ((b.power_w.iloc[0] - p0) / b.tflops.iloc[0])
    return f


def rho_onecap(dt, fr, g, nc, p0):
    """以最低上限一點校正；該點必須 power-bound 且能反查原生曲線。"""
    r = g.sort_values("cap").iloc[0]
    pc = native_cap(nc, r.perf)
    if np.isnan(pc):
        return None
    return (r.cap - p0) / (pc - p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-mi350", type=float, default=None)
    ap.add_argument("--p0-gh200", type=float, default=None)
    a = ap.parse_args()
    out, summ = [], []
    plats = {"MI350X": (mi350_curves(), a.p0_mi350), "GH200": (gh200_curves(), a.p0_gh200)}
    fixed = mi350_fixed()
    for plat, (cur, p0_given) in plats.items():
        methods = [("onecap", rho_onecap)] + ([("fixed", rho_fixed(fixed))] if plat == "MI350X" else [])
        for mname, fn in methods:
            grid = [p0_given] if p0_given is not None else list(range(0, 351, 25))
            best = None
            for p0 in grid:
                d = predict(cur, fn, p0)
                # onecap 的校正點本身不計入誤差
                ev = d if mname == "fixed" else d[d.cap != d.groupby(["dtype", "frac"]).cap.transform("min")]
                mae = ev.err_pts.abs().mean()
                summ.append(dict(platform=plat, method=mname, p0=p0, mae_pts=mae, max_pts=ev.err_pts.abs().max(), n=len(ev)))
                if best is None or mae < best[0]:
                    best = (mae, p0, d)
            d = best[2].assign(platform=plat, method=mname, p0=best[1])
            out.append(d)
        if plat == "MI350X":
            grid = [p0_given] if p0_given is not None else list(range(0, 351, 25))
            best = None
            for p0 in grid:
                d = predict_stretch(cur, fixed, p0)
                mae = d.err_pts.abs().mean()
                summ.append(dict(platform=plat, method="fixed+stretch", p0=p0, mae_pts=mae, max_pts=d.err_pts.abs().max(),
                                 n=len(d)))
                if best is None or mae < best[0]:
                    best = (mae, p0, d)
            out.append(best[2].assign(platform=plat, method="fixed+stretch", p0=best[1]))
    pred = pd.concat(out, ignore_index=True)
    s = pd.DataFrame(summ)
    pred.to_csv(f"{RES}/model_pred.csv", index=False)
    s.to_csv(f"{RES}/model_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.loc[s.groupby(["platform", "method"]).mae_pts.idxmin()].to_string(index=False))
    print(s[s.p0 == 0].to_string(index=False))
    print(pred.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
