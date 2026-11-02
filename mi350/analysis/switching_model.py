"""Switching-aware prediction (review round 6, N3): from operand bit flips to the truncated GEMM's speedup under caps,
with no measurement of the truncated variant.

1. Fit, at a locked 1300 MHz (run mechb1300), the dynamic-power ratio R = (P - P0)/(P_native - P0) and the effective-clock
   ratio sigma of the 12 layouts that are NOT plain zero-fill truncation (sorted, shuffled, krep, mrep, ones, const) on the
   flip changes along k and across rows and their product.
2. Predict R and sigma for plain truncation to b kept bits from the flip statistics of its operands (analysis/flipstats.py).
3. Feed them into the model of Section V (P0 = idle 262 W, native curve of node n02) and compare the predicted speedups
   with the measured cap sweep (e1n02_1029_0502), next to the model calibrated by locked-clock runs of the truncated
   variants themselves (results/model_eval_points.csv, method idle).
usage: switching_model.py → results/switching_model.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import load  # noqa: E402
from flipstats import flips  # noqa: E402
from power_model import mi350_curves, mi350_fixed, native_perf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs")
P0 = 262.0
RID = sorted(f[:-4] for f in os.listdir(LOG) if f.startswith("mechb1300_") and f.endswith(".out"))[0]


def main():
    d = load([RID], LOG)
    d = d[d.dtype == "fp64"]
    lay = d.groupby(["frac", "low"]).agg(P=("power_acc_w", "mean"), clk=("gfxclk", "mean"),
                                          tog=("tog", "mean"), tox=("tox", "mean")).reset_index()
    nat = lay[(lay.frac == 52) & (lay.low == "zero")].iloc[0]
    lay["R"] = (lay.P - P0) / (nat.P - P0)
    lay["sigma"] = lay.clk / nat.clk
    lay["dk"], lay["dx"] = lay.tog - nat.tog, lay.tox - nat.tox
    train = lay[lay.low != "zero"]
    X = lambda t: np.column_stack([np.ones(len(t)), t.dk, t.dx, t.dk * t.dx])  # noqa: E731
    cR, *_ = np.linalg.lstsq(X(train), train.R, rcond=None)
    cS, *_ = np.linalg.lstsq(X(train), train.sigma, rcond=None)
    # predicted R, sigma for plain truncation from its operands' flips
    fixed = mi350_fixed()
    fx = fixed[fixed.dtype == "fp64"].set_index("frac")
    rows = []
    for b in (0, 12, 20, 27, 36, 44, 52):
        tog, tox = flips(b)
        t = pd.DataFrame(dict(dk=[tog - nat.tog], dx=[tox - nat.tox]))
        Rp, Sp = float((X(t) @ cR)[0]), float((X(t) @ cS)[0])
        Rm = Sm = np.nan
        if b in fx.index:
            Rm = (fx.loc[b, "power_w"] - P0) / (fx.loc[52, "power_w"] - P0)
            Sm = fx.loc[b, "gfxclk"] / fx.loc[52, "gfxclk"]
        rows.append(dict(bits=b, tog=tog, tox=tox, R_pred=Rp, R_meas=Rm, sigma_pred=Sp, sigma_meas=Sm))
    rs = pd.DataFrame(rows)
    # speedups under caps from predicted R, sigma
    cur = mi350_curves()
    nc = cur[(cur.dtype == "fp64") & (cur.frac == 52)].sort_values("cap")
    ev = pd.read_csv(f"{RES}/model_eval_points.csv")
    ev = ev[(ev.method == "idle") & (ev.platform == "MI350X") & (ev.dtype == "fp64")]
    pts = []
    for r in rs.itertuples():
        g = cur[(cur.dtype == "fp64") & (cur.frac == r.bits)]
        if g.empty or r.bits == 52:
            continue
        for c in g.itertuples():
            pn = native_perf(nc, c.cap)
            pp = P0 + (c.cap - P0) / r.R_pred
            base = native_perf(nc, pp)
            if np.isnan(base):
                last = nc.iloc[-2:]
                base = last.perf.iloc[1] + (last.perf.iloc[1] - last.perf.iloc[0]) / (last.cap.iloc[1] - last.cap.iloc[0]) * (pp - last.cap.iloc[1])
            pred = min(r.sigma_pred * base, g.perf.max())
            lc = ev[(ev.frac == r.bits) & (ev.cap == c.cap)]
            pts.append(dict(bits=r.bits, cap=c.cap, meas=100 * (c.perf / pn - 1), pred_flips=100 * (pred / pn - 1),
                            pred_lockclk=float(lc.pred_gain.iloc[0]) if len(lc) else np.nan))
    p = pd.DataFrame(pts)
    p["err_flips"] = p.pred_flips - p.meas
    p["err_lockclk"] = p.pred_lockclk - p.meas
    rs.to_csv(f"{RES}/switching_model_R.csv", index=False)
    p.to_csv(f"{RES}/switching_model.csv", index=False)
    pd.set_option("display.width", 200)
    print("fit on", len(train), "non-zero-fill layouts; R coef", np.round(cR, 4), "sigma coef", np.round(cS, 5))
    print(rs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(p.groupby("bits")[["err_flips", "err_lockclk"]].agg(lambda x: f"{x.abs().mean():.1f}/{x.abs().max():.1f}").to_string())
    print(f"all: flips-based mean |err| {p.err_flips.abs().mean():.2f} (max {p.err_flips.abs().max():.2f}); "
          f"locked-clock-calibrated {p.err_lockclk.abs().mean():.2f} (max {p.err_lockclk.abs().max():.2f}) over {len(p)} points")


if __name__ == "__main__":
    main()
