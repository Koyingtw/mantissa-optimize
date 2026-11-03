"""功耗模型的外部驗證（回應審稿意見 2）：模型只用 GEMM 微基準建立與評估；這裡把它套到兩個沒有參與的應用。

對 DF-MP2 與 GP（node n02）：
  1. 鎖頻 1300 MHz、1000 W 的執行（scripts/lockclk_apps.sh → mp2lock1300_*、gplock1300_*）給出每個變體的每卡平均功耗
     與有效時脈；R = (P_r − P0)/(P_n − P0)（動態功耗比），σ = clk_r/clk_n。P0 = n02 的閒置功耗 262 W（不擬合）。
  2. 原生程式在 500/650/800/1000 W 的效能曲線 p_n(P) = 1/wall（mp2n02、gpn02）。
  3. 預測 p_r(P) = σ·p_n(P0 + (P − P0)/R)：在原生點之間線性內插，超過 1000 W 時以最後一段外插。
  4. 與同一節點實測的加速比較。
usage: model_apps.py → results/model_apps.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
P0 = 262.0


def latest(prefix):
    f = sorted(x for x in os.listdir(RES) if x.startswith(prefix) and x.endswith("_summary.csv"))
    if not f:
        raise SystemExit(f"no {prefix}*_summary.csv in results/")
    return pd.read_csv(f"{RES}/{f[-1]}"), f[-1]


def native_perf(curve, cap):
    c = curve.sort_values("cap")
    if cap <= c.cap.max():
        return float(np.interp(cap, c.cap, c.perf))
    a, b = c.iloc[-2], c.iloc[-1]
    return b.perf + (b.perf - a.perf) / (b.cap - a.cap) * (cap - b.cap)


def main():
    rows = []
    for app, lock_prefix, sweep_prefix, variants in [("DF-MP2", "mp2lock1300_", "mp2n02_", ["t32n", "t20n"]),
                                                     ("GP", "gplock1300_", "gpn02_", ["t36n", "t28n"])]:
        lock, lf = latest(lock_prefix)
        sweep, sf = latest(sweep_prefix)
        lock = lock[lock.cap == 1000].set_index("variant")
        n = lock.loc["native"]
        curve = sweep[sweep.variant == "native"].assign(perf=lambda d: 1.0 / d.wall)[["cap", "perf"]]
        for v in variants:
            r = lock.loc[v]
            R = (r.power_w - P0) / (n.power_w - P0)
            sigma = r.gfxclk / n.gfxclk
            # review round 9: for GP the whole-program ratios come from the in-program energy counter (gp_energy.py);
            # the earlier sampler-based locked run (gplock1300) gave 0.96 / 0.93 with a 0.91-0.95 spread across GPUs
            # that the in-program measurement (0.875-0.891) does not show
            # review round 10: both applications from the in-program energy counter of scripts/xval.sh, whose truncated
            # and native windows ran at equal temperature (the gplock1300 GP run had them 7 degC apart, which leakage at
            # the locked clock turns into an apparent R closer to 1); sampler and counter agree there within 0.001
            ge = f"{RES}/{'gp' if app == 'GP' else 'mp2'}_energy.csv"
            if os.path.exists(ge):
                e = pd.read_csv(ge, index_col=0)
                R = float(e.loc[v, "R_whole"])
                sigma = float(e.loc["native", "wall"] / e.loc[v, "wall"])
                lf = os.path.basename(ge)
            for cap in sorted(sweep.cap.unique()):
                pn = native_perf(curve, cap)
                pred = sigma * native_perf(curve, P0 + (cap - P0) / R)
                meas = float(sweep[(sweep.cap == cap) & (sweep.variant == v)].speedup_pct.iloc[0])
                rows.append(dict(app=app, variant=v, cap=cap, R=R, sigma=sigma, lock_power_native=n.power_w,
                                 lock_power_variant=r.power_w, lock_clk_native=n.gfxclk, lock_clk_variant=r.gfxclk,
                                 pred_pct=100 * (pred / pn - 1), meas_pct=meas, err_pts=100 * (pred / pn - 1) - meas,
                                 lock_file=lf, sweep_file=sf))
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/model_apps.csv", index=False)
    pd.set_option("display.width", 200)
    print(d[["app", "variant", "cap", "R", "sigma", "pred_pct", "meas_pct", "err_pts"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"mean |err| {d.err_pts.abs().mean():.2f} pts, max {d.err_pts.abs().max():.2f} pts over {len(d)} points")
    # review round 10: run-to-run spread of GP's whole-program R. The same in-program counter in job 426
    # (gp_energy_1103_2220.csv, no temperature log; its unchanged non-GEMM phases drew 3-7 % less in the truncated
    # variants) and the sampler-based gplock1300 run (truncated variants 7 degC hotter) give other R; errors with each
    sweep, _ = latest("gpn02_")
    curve = sweep[sweep.variant == "native"].assign(perf=lambda d: 1.0 / d.wall)[["cap", "perf"]]
    lock, lf = latest("gplock1300_")
    lock = lock[lock.cap == 1000].set_index("variant")
    alts = {lf: {v: ((lock.loc[v].power_w - P0) / (lock.loc["native"].power_w - P0),
                     lock.loc[v].gfxclk / lock.loc["native"].gfxclk) for v in ("t36n", "t28n")}}
    for f in ("gp_energy_1103_2220.csv", "gp_energy.csv"):
        if os.path.exists(f"{RES}/{f}"):
            e = pd.read_csv(f"{RES}/{f}", index_col=0)
            alts[f] = {v: (e.loc[v, "R_whole"], e.loc["native", "wall"] / e.loc[v, "wall"]) for v in ("t36n", "t28n")}
    for f, rs in alts.items():
        err = []
        for v, (R, sigma) in rs.items():
            for cap in sorted(sweep.cap.unique()):
                pn = native_perf(curve, cap)
                pred = sigma * native_perf(curve, P0 + (cap - P0) / R)
                err.append(100 * (pred / pn - 1) - float(sweep[(sweep.cap == cap) & (sweep.variant == v)].speedup_pct.iloc[0]))
        err = np.array(err)
        print(f"GP R from {f}: t36n {rs['t36n'][0]:.3f} t28n {rs['t28n'][0]:.3f}; mean |err| {np.abs(err).mean():.2f} "
              f"max {np.abs(err).max():.2f} mean signed {err.mean():+.2f} over {len(err)} points")


if __name__ == "__main__":
    main()
