#!/usr/bin/env python3
"""Figures 6.1-6.4 of the Traditional-Chinese project report (MI350X chapter), drawn only from mi350/results/ and
gh200/results/; fonts and style come from gh200/analysis/report_figs_zh.py so both chapters look the same.

usage: report_figs_zh_mi350.py <font dir> [outdir]      (default outdir: <repo>/paper/zh_report/figs)
  CHECK=1 / PREVIEW=<dir> as in report_figs_zh.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "gh200", "analysis"))
sys.path.insert(0, HERE)
import report_figs_zh as z  # noqa: E402
from paper_figures_mi350 import mi350_windows  # noqa: E402

plt = z.plt
RES = os.path.join(HERE, "..", "results")
GH = os.path.join(HERE, "..", "..", "gh200", "results")
CAP_GPU = "每張 GPU 的功耗上限（W）"


def save(fig, name):
    path = os.path.join(z.OUT, name)
    fig.savefig(path, metadata={"Creator": "mi350/analysis/report_figs_zh_mi350.py", "CreationDate": None})
    if os.environ.get("CHECK"):
        z.check_layout(fig, name)
    if os.environ.get("PREVIEW"):
        os.makedirs(os.environ["PREVIEW"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.replace(".pdf", ".png")), dpi=200)
    plt.close(fig)
    print(f"已寫出 {os.path.relpath(path)}")


def fig_gemm_caps():
    """6.1 MI350X GEMM speedup vs package cap. Window level (8 GPUs averaged per window): node n02 mean with bars over
    its two windows (e1n02_1029_0502), open markers node n01 (e1_1028_1709)."""
    w2, w1 = mi350_windows("e1n02_1029_0502"), mi350_windows("e1_1028_1709")
    series = [("fp64", 36, "FP64 保留 36 位", z.BLUE, "o", 0), ("fp64", 27, "FP64 保留 27 位", z.BLUE_L, "s", -8),
              ("bf16", 0, "BF16 保留 0 位", z.ORANGE, "^", 8), ("fp8", 0, "FP8 保留 0 位", z.GREY_D, "D", 0)]
    fig, ax = z.fig_axes(z.FULL, 6.6, 1.35, 0.3, 1.0, 0.25)
    for dt, fr, lab, c, mk, dx in series:
        g = w2[(w2.dtype == dt) & (w2.frac == fr)].groupby("cap").gain.agg(["mean", "min", "max"]).reset_index()
        ax.errorbar(g["cap"] + dx, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], color=c, marker=mk,
                    ms=4, lw=1.3, elinewidth=0.8, capthick=0.8, label=lab, zorder=3)
        h = w1[(w1.dtype == dt) & (w1.frac == fr)].groupby("cap").gain.mean().reset_index()
        ax.scatter(h["cap"] + dx, h["gain"], marker=mk, s=16, facecolors="white", edgecolors=c, lw=0.8, zorder=4)
    ax.scatter([], [], marker="o", s=16, facecolors="white", edgecolors=z.GREY, lw=0.8, label="空心：節點 n01")
    ax.set_xlim(370, 1030)
    ax.set_xticks(range(400, 1001, 100))
    ax.set_ylim(0, 70)
    ax.set_xlabel(CAP_GPU)
    ax.set_ylabel(z.REL_GAIN)
    h, l = ax.get_legend_handles_labels()
    order = [i for i, t in enumerate(l) if "n01" not in t] + [i for i, t in enumerate(l) if "n01" in t]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper right", ncol=2)
    save(fig, "fig6_01_gemm_caps.pdf")


def fig_model():
    """6.2 Predicted vs measured GEMM speedup, P0 = measured idle power (model_eval_points.csv, method idle)."""
    d = pd.read_csv(f"{RES}/model_eval_points.csv")
    d = d[d.method == "idle"]
    fig, ax = z.fig_axes(z.HALF + 1.5, 7.0, 1.2, 0.3, 1.0, 0.25)
    for plat, lab, c, mk in [("GH200", "GH200（$P_0$ = 閒置 174 W）", z.BLUE, "o"),
                             ("MI350X", "MI350X（$P_0$ = 閒置 262 W）", z.ORANGE, "s")]:
        g = d[d.platform == plat]
        e = (g.pred_gain - g.meas_gain).abs()
        ax.scatter(g.meas_gain, g.pred_gain, s=12, color=c, marker=mk, alpha=0.85, lw=0,
                   label=f"{lab}\n平均誤差 {e.mean():.1f}、最大 {e.max():.1f} 個百分點", zorder=3)
    top = 10 * np.ceil(max(d.meas_gain.max(), d.pred_gain.max()) / 10)
    lim = [0, top]
    ax.set_xticks(np.arange(0, top + 1, 10))
    ax.set_yticks(np.arange(0, top + 1, 10))
    ax.plot(lim, lim, color=z.GREY_L, lw=0.8, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("實測加速（%）")
    ax.set_ylabel("模型預測的加速（%）")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "fig6_02_model.pdf")


def fig_apps():
    """6.3 MI350X applications: pooled speedup with 95% t intervals over independent units (apps_units.csv)."""
    u = pd.read_csv(f"{RES}/apps_units.csv")
    apps = [("HPL, 36 bits", "FP64 HPL，36 位", z.BLUE_D), ("HPL-MxP, 3 bits", "HPL-MxP，3 位", z.BLUE_L),
            ("GP, 28 bits", "高斯過程，28 位", z.GREY), ("DF-MP2, 20 bits", "DF-MP2，20 位", z.ORANGE)]
    caps = [1000, 800, 650, 500]
    fig, ax = z.fig_axes(z.FULL, 6.2, 1.35, 0.3, 0.8, 0.25)
    w = 0.8 / len(apps)
    for i, (key, lab, c) in enumerate(apps):
        g = u[u.app == key].set_index("cap").reindex(caps)
        x = np.arange(len(caps)) + (i - (len(apps) - 1) / 2) * w
        lo, hi = int(g.units.min()), int(g.units.max())
        nlab = f"{lo}" if lo == hi else f"{lo}–{hi}"
        ax.bar(x, g["mean"], w * 0.92, color=c, label=f"{lab}（$n$ = {nlab}）", yerr=g.ci95,
               error_kw={"lw": 0.7, "capsize": 1.5, "capthick": 0.7, "ecolor": z.INK2}, zorder=3)
        for xi, v, e in zip(x, g["mean"], g.ci95):
            ax.text(xi, v + e + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=z.INK2)
    ax.set_xticks(np.arange(len(caps)), [f"{c} W" for c in caps])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 25)
    ax.set_ylabel(z.REL_GAIN)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig6_03_apps.pdf")


def fig_crossover():
    """6.4 MI350X: truncation vs GEMMul8 over native FP64 as a function of k at three caps (xover_1029_1508)."""
    s = pd.read_csv(f"{RES}/xover_1029_1508_summary.csv")
    series = [("fp64", 35, "截斷，保留 35 位（誤差 $7.8{\\times}10^{-12}$）", z.BLUE, "o", "-"),
              ("fp64", 27, "截斷，保留 27 位（$2.0{\\times}10^{-9}$）", z.BLUE_L, "s", "--"),
              ("g8", 10, "GEMMul8，10 個模數（$1.0{\\times}10^{-10}$）", z.ORANGE_M, "^", "--"),
              ("g8", 12, "GEMMul8，12 個模數（$(4$–$8){\\times}10^{-13}$）", z.ORANGE, "D", "-")]
    top = 25 * np.ceil(s[s.dtype == "g8"].speedup_pct.max() / 25 + 0.2)
    fig = plt.figure(figsize=(z.FULL * z.CM, 7.4 * z.CM))
    axes = [fig.add_axes([(1.35 + i * 4.75) / z.FULL, 2.3 / 7.4, 4.25 / z.FULL, 4.7 / 7.4]) for i in range(3)]
    for i, (ax, cap) in enumerate(zip(axes, (1000, 700, 500))):
        for dt, fr, lab, c, mk, ls in series:
            g = s[(s.dtype == dt) & (s.frac == fr) & (s.cap == cap)].sort_values("k")
            ax.plot(g.k, g.speedup_pct, color=c, marker=mk, ms=3.5, lw=1.2, ls=ls, label=lab, zorder=3)
        ax.axhline(0, color=z.GREY, lw=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096, 8192], ["1024", "2048", "4096", "8192"])
        ax.minorticks_off()
        ax.set_ylim(-60, top)
        ax.text(0.04, 0.96, f"{cap} W", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        ax.set_xlabel("內積長度 $k$（$m = n = 32768$）")
        if i:
            ax.set_yticklabels([])
    axes[0].set_ylabel("相對原生 FP64 的加速（%）")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
    save(fig, "fig6_04_crossover.pdf")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    z.OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "paper", "zh_report", "figs")
    os.makedirs(z.OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        z.setup_style(z.setup_fonts(sys.argv[1], tmp))
        for f in (fig_gemm_caps, fig_model, fig_apps, fig_crossover):
            f()


if __name__ == "__main__":
    main()
