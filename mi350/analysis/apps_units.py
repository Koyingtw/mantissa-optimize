"""MI350X 應用結果的統一不確定度：以真正獨立的單位計算，兩節點合併，並列溫度校正值（回應審稿意見 1、3）。

單位：HPL／HPL-MxP = 一次整節點執行（比值以同一 ABBA 區塊的原生平均為分母）；GP／DF-MP2 = 一輪（8 張 GPU 先平均，
比值為同輪原生的平均 wall ÷ 變體的平均 wall）。95% t 區間的自由度 = 單位數 − 1。
溫度校正：HPL／HPL-MxP 用 temp_adjust.py 的逐節點結果（每次執行的 8 卡平均溫度）；GP／DF-MP2 用每張 GPU 的每次執行，
log(1/wall) ~ 變體 + GPU 固定效果 + 該次的平均 junction 溫度，逐節點估計。
usage: apps_units.py → results/apps_units.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_summary import BDF  # noqa: E402,F401
from smi import load_smi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(HERE, "..", "logs")

APPS = {  # app: (variant, native, [(node, result id, log dir, kind)])
    "HPL, 36 bits": ("36:36", "53:53", [("n01", "hpl_1029_0336", LOG, "run"), ("n02", "hpln02_1029_1045", f"{LOG}/n02", "run")]),
    "HPL-MxP, 3 bits": ("3", "native", [("n01", "mxp_1028_2359", LOG, "run"), ("n02", "mxpn02_1029_0810", f"{LOG}/n02", "run")]),
    "GP, 28 bits": ("t28n", "native", [("n01", "gp_1029_0731", LOG, "round"), ("n02", "gpn02_1030_0521", f"{LOG}/n02", "round")]),
    "GP, 36 bits": ("t36n", "native", [("n01", "gp_1029_0731", LOG, "round"), ("n02", "gpn02_1030_0521", f"{LOG}/n02", "round")]),
    "DF-MP2, 20 bits": ("t20n", "native", [("n01", "mp2_1029_1607", LOG, "round"), ("n02", "MP2N02", f"{LOG}/n02", "round")]),
    "DF-MP2, 32 bits": ("t32n", "native", [("n01", "mp2_1029_1607", LOG, "round"), ("n02", "MP2N02", f"{LOG}/n02", "round")]),
    # GEMMul8 (warm pass only): n01 from the separate emulation job, n02 from the replication job
    "DF-MP2, GEMMul8 12": ("g8m12", "native", [("n01", "mp2emu_1029_1741", LOG, "round"), ("n02", "MP2N02", f"{LOG}/n02", "round")]),
}


def tci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def run_units(rid, var, nat):
    d = pd.read_csv(f"{RES}/{rid}_raw.csv")
    d = d[d.passed & (d.pass_ != "W")]
    d["block"] = d["block"] if "block" in d else d.pass_.str[1:]
    out = {}
    for cap, g in d.groupby("cap"):
        r = []
        for b, h in g.groupby("block"):
            n = h[h.variant.astype(str) == nat].tflops.mean()
            r += list(100 * (h[h.variant.astype(str) == var].tflops / n - 1))
        out[cap] = r
    return out


def round_units(rid, var, nat):
    d = pd.read_csv(f"{RES}/{rid}_raw.csv")
    if "cold" in d:
        d = d[~d.cold]
    out = {}
    for cap, g in d.groupby("cap"):
        out[cap] = [100 * (h[h.variant == nat].wall.mean() / h[h.variant == var].wall.mean() - 1)
                    for _, h in g.groupby("round")]
    return out


def round_tadj(rid, logdir, var, nat):
    """per GPU-run regression with GPU fixed effects and mean junction temperature, per cap."""
    d = pd.read_csv(f"{RES}/{rid}_raw.csv")
    if "cold" in d:
        d = d[~d.cold]
    d = d[d.variant.isin([var, nat])]
    smi = load_smi(f"{logdir}/{rid}_smi.csv")
    # t_begin/t_end are not in the GP/MP2 raw files; recover them from the log
    import re
    tb = {}
    for line in open(f"{logdir}/{rid}.out"):
        if line.startswith(("RESULT_GP", "RESULT_MP2")):
            kv = dict(re.findall(r"(\w+)=([^\s,\[]+)", line.replace("tag=", "")))
            tb.setdefault((int(kv["cap"]), int(kv["round"]), kv["variant"], kv["pass"], int(kv["gpu"])), []).append(
                (float(kv["t_begin"]), float(kv["t_end"])))
    out = {}
    for cap, g in d.groupby("cap"):
        rows = []
        for r in g.itertuples():
            # raw files carry the amd-smi index; the log carries the HIP device index — match on time windows instead
            cands = [v for (c, rd, vv, ps, dev), v in tb.items() if c == cap and rd == r.round and vv == r.variant
                     and ps == r.pass_]
            T = np.nan
            for lst in cands:
                for t0, t1 in lst:
                    w = smi[(smi.gpu == r.gpu) & (smi.t >= t0) & (smi.t <= t1)]
                    if len(w) > 3 and abs((t1 - t0) - r.wall) < 0.5:
                        T = w.t_hot.mean()
            rows.append((r.gpu, r.variant, r.wall, T))
        x = pd.DataFrame(rows, columns=["gpu", "variant", "wall", "T"]).dropna()
        if x.variant.nunique() < 2:
            continue
        X = [np.ones(len(x)), (x.variant == var).astype(float)]
        X += [(x.gpu == k).astype(float) for k in sorted(x.gpu.unique())[1:]] + [x["T"].values]
        b, *_ = np.linalg.lstsq(np.column_stack(X), -np.log(x.wall.values), rcond=None)
        out[cap] = 100 * (np.exp(b[1]) - 1)
    return out


def main():
    mp2n02 = sorted(f[:-8] for f in os.listdir(RES) if f.startswith("mp2n02_") and f.endswith("_raw.csv"))
    rows = []
    for app, (var, nat, nodes) in APPS.items():
        per_node = {}
        for node, rid, logdir, kind in nodes:
            if rid == "MP2N02":
                if not mp2n02:
                    continue
                rid = mp2n02[-1]
            u = run_units(rid, var, nat) if kind == "run" else round_units(rid, var, nat)
            if kind == "run":
                t = pd.read_csv(f"{RES}/{rid}_tempadj.csv")
                ta = {r.cap: r.adj_pct for r in t[t.variant.astype(str) == var].itertuples()}
            else:
                ta = round_tadj(rid, logdir, var, nat)
            per_node[node] = (u, ta, kind)
        caps = sorted(set().union(*[set(v[0]) for v in per_node.values()]), reverse=True)
        for cap in caps:
            allu = sum((v[0].get(cap, []) for v in per_node.values()), [])
            row = dict(app=app, cap=cap, unit=list(per_node.values())[0][2], units=len(allu), mean=np.mean(allu),
                       ci95=tci(allu))
            for node, (u, ta, _) in per_node.items():
                row[f"{node}_raw"] = np.mean(u.get(cap, [np.nan]))
                row[f"{node}_n"] = len(u.get(cap, []))
                row[f"{node}_tadj"] = ta.get(cap, np.nan)
            rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(f"{RES}/apps_units.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
