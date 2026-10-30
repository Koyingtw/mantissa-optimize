"""MI350X 主要結果的保守不確定度（回應「同機箱 8 張 GPU、同 job 的重複不是獨立樣本」）。

E1 GEMM：獨立單位改為「量測窗」：同一 pass、同一設定的 8 張 GPU 先平均（它們同時在同一機箱裡跑），比值以同 pass 的原生
窗口為分母；每節點每上限 2 個窗口，兩節點合計 4 個，95% t 區間（自由度 3）。另做溫度校正：每節點每上限內，
log(TFLOP/s) ~ 設定 + GPU 固定效果 + 該 GPU 窗口平均 junction 溫度。
HPL／HPL-MxP：沿用 hpl_summary（每次執行已是全節點）與 temp_adjust 的逐節點結果，並列兩節點。
GP／DF-MP2：單位改為「輪」（8 GPU 平均），逐節點列出。
usage: robust_summary.py → results/robust_e1.csv、results/robust_apps.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import NATIVE, load  # noqa: E402
from smi import load_smi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs")
E1 = [("n01", "e1_1028_1709", LOG), ("n02", "e1n02_1029_0502", os.path.join(LOG, "n02"))]
KEY = [("fp64", 36), ("fp64", 27), ("fp64", 0), ("bf16", 3), ("bf16", 0), ("fp8", 0)]


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def e1_temperature(d, smi):
    """per-GPU 窗口平均 junction 溫度（e1_summary 只有最大值）。"""
    T = []
    for r in d.itertuples():
        w = smi[(smi.gpu == r.gpu) & (smi.t >= r.t_begin) & (smi.t <= r.t_end)]
        T.append(w.t_hot.mean())
    return d.assign(T=T)


def robust_e1():
    rows = []
    frames = []
    for node, rid, logdir in E1:
        d = load([rid], logdir)
        d = d[(d.low == "zero")]
        d = e1_temperature(d, load_smi(f"{logdir}/{rid}_smi.csv"))
        frames.append(d.assign(node=node))
    d = pd.concat(frames)
    caps = sorted(set(d[d.node == "n01"].cap) & set(d[d.node == "n02"].cap), reverse=True)
    for dt, fr in KEY:
        for cap in caps:
            win, adj = [], {}
            for node, g in d[(d.cap == cap) & (d.dtype == dt)].groupby("node"):
                for pas, h in g.groupby("pass_"):
                    v = h[h.frac == fr].tflops.mean()
                    n = h[h.frac == NATIVE[dt]].tflops.mean()
                    if np.isfinite(v) and np.isfinite(n):
                        win.append(v / n - 1)
                # 溫度校正：GPU 固定效果 + 溫度
                gg = g[g.frac.isin([fr, NATIVE[dt]])]
                if gg.frac.nunique() < 2:
                    continue
                X = [np.ones(len(gg)), (gg.frac == fr).astype(float)]
                X += [(gg.gpu == k).astype(float) for k in sorted(gg.gpu.unique())[1:]]
                X += [gg["T"].values]
                X = np.column_stack(X)
                b, *_ = np.linalg.lstsq(X, np.log(gg.tflops.values), rcond=None)
                adj[node] = 100 * (np.exp(b[1]) - 1)
            if len(win) < 2:
                continue
            w = 100 * np.array(win)
            raw = {node: 100 * (d[(d.node == node) & (d.cap == cap) & (d.dtype == dt) & (d.frac == fr)].tflops.mean() /
                                d[(d.node == node) & (d.cap == cap) & (d.dtype == dt) & (d.frac == NATIVE[dt])]
                                .tflops.mean() - 1) for node in ("n01", "n02")}
            rows.append(dict(dtype=dt, frac=fr, cap=cap, windows=len(w), mean=w.mean(), ci95_windows=tci(w),
                             n01=raw["n01"], n02=raw["n02"], n01_tadj=adj.get("n01", np.nan),
                             n02_tadj=adj.get("n02", np.nan)))
    r = pd.DataFrame(rows)
    r.to_csv(f"{RES}/robust_e1.csv", index=False)
    return r


def robust_apps():
    rows = []
    for app, var, files in [("HPL 36 bits", "36:36", [("n01", "hpl_1029_0336"), ("n02", "hpln02_1029_1045")]),
                            ("HPL-MxP 3 bits", "3", [("n01", "mxp_1028_2359"), ("n02", "mxpn02_1029_0810")])]:
        for node, rid in files:
            s = pd.read_csv(f"{RES}/{rid}_summary.csv")
            t = pd.read_csv(f"{RES}/{rid}_tempadj.csv")
            for r in s[s.variant.astype(str) == var].itertuples():
                ta = t[(t.cap == r.cap) & (t.variant.astype(str) == var)]
                rows.append(dict(app=app, node=node, cap=r.cap, raw=r.speedup_pct, lo=r.speedup_lo, hi=r.speedup_hi,
                                 tadj=float(ta.adj_pct.iloc[0]) if len(ta) else np.nan, unit="run (8 GPUs)"))
    for app, var, rid, kind in [("GP 28 bits", "t28n", "gp_1029_0731", "gp"), ("DF-MP2 20 bits", "t20n", "mp2_1029_1607", "mp2")]:
        raw = pd.read_csv(f"{RES}/{rid}_raw.csv")
        raw = raw[~raw.get("cold", False)] if "cold" in raw else raw
        for cap, g in raw.groupby("cap"):
            per = []
            for rnd, h in g.groupby("round"):
                nat = h[h.variant == "native"].wall.mean()
                per.append(100 * (nat / h[h.variant == var].wall.mean() - 1))
            per = np.array(per)
            rows.append(dict(app=app, node="n01", cap=cap, raw=per.mean(), lo=per.mean() - tci(per),
                             hi=per.mean() + tci(per), tadj=np.nan, unit=f"round (8 GPUs), n={len(per)}"))
    r = pd.DataFrame(rows).sort_values(["app", "cap", "node"], ascending=[True, False, True])
    r.to_csv(f"{RES}/robust_apps.csv", index=False)
    return r


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(robust_e1().to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(robust_apps().to_string(index=False, float_format=lambda x: f"{x:.2f}"))
