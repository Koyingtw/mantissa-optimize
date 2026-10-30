"""溫度校正：在同一個上限內以 log(性能) ~ 變體 + 平均 junction 溫度 做最小平方法，回報校正後的相對原生速度。

用途：MI350X 機箱風扇依前一次執行的溫度振盪，讓相同設定的執行相差 10–15 °C、性能差約 0.4–0.5%/°C（實驗日誌 00:55）。
ABBA 能抵消線性漂移，抵消不了這種非線性的散熱狀態；以溫度為共變量可以估計「溫度相同時」的效果。
usage: temp_adjust.py <result id> [--logdir logs] [--kind hpl]
  kind=hpl：results/<id>_raw.csv（hpl_summary.py 的輸出），每次執行一列，溫度取窗口內（去頭 30 s、去尾 15 s）8 卡平均 t_hot
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from smi import load_smi  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def adjust(d, native, value="tflops", temp="T"):
    """d: 一個上限內的執行；回傳 {variant: (raw %, adjusted %, 95% half-width, temp coef %/°C)}。"""
    d = d.assign(y=np.log(d[value]))
    vs = [v for v in d.variant.astype(str).unique() if v != native]
    X = np.column_stack([np.ones(len(d))] + [(d.variant.astype(str) == v).astype(float) for v in vs] + [d[temp]])
    b, *_ = np.linalg.lstsq(X, d.y, rcond=None)
    dof = len(d) - X.shape[1]
    out = {}
    if dof <= 0:
        return out
    r = d.y - X @ b
    cov = (r ** 2).sum() / dof * np.linalg.inv(X.T @ X)
    base = d[d.variant.astype(str) == native].y.mean()
    for i, v in enumerate(vs):
        raw = np.exp(d[d.variant.astype(str) == v].y.mean() - base) - 1
        out[v] = (100 * raw, 100 * (np.exp(b[i + 1]) - 1), 196 * np.sqrt(cov[i + 1, i + 1]), 100 * b[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--logdir", default="logs")
    a = ap.parse_args()
    d = pd.read_csv(f"{RES}/{a.id}_raw.csv")
    d = d[d.passed & (d.pass_ != "W")]
    s = load_smi(f"{a.logdir}/{a.id}_smi.csv")
    d["T"] = [s[(s.t >= r.t_begin + 30) & (s.t <= r.t_end - 15)].t_hot.mean() for r in d.itertuples()]
    native = "native" if (d.bench == "mxp").all() else "53:53"
    rows = []
    for cap, g in d.groupby("cap"):
        for v, (raw, adj, hw, tc) in adjust(g, native).items():
            rows.append(dict(cap=cap, variant=v, n=int((g.variant.astype(str) == v).sum()), raw_pct=raw, adj_pct=adj,
                             adj_hw=hw, temp_coef_pct_per_C=tc, T_native=g[g.variant.astype(str) == native]["T"].mean(),
                             T_variant=g[g.variant.astype(str) == v]["T"].mean()))
    o = pd.DataFrame(rows).sort_values(["cap", "variant"], ascending=[False, True])
    o.to_csv(f"{RES}/{a.id}_tempadj.csv", index=False)
    print(o.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
