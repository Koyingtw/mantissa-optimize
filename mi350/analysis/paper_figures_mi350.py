"""Two-platform figures for the paper (paper/figs/mi350_*.pdf), drawn only from summarized CSV files:
mi350/results/*.csv and gh200/results/*.csv, in the style of gh200/analysis/paper_figures.py.

usage: paper_figures_mi350.py [outdir]      (default ../paper/figs relative to mi350/; PREVIEW=<dir> also writes PNGs)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
GH = os.path.join(HERE, "..", "..", "gh200", "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.45
WIDE = 7.0
C = ["#2a6f97", "#d1495b", "#edae49", "#66a182", "#8d6cab", "#555555"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    if os.environ.get("PREVIEW"):
        os.makedirs(os.environ["PREVIEW"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.replace(".pdf", ".png")), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def mi350_e1():
    """MI350X E1 on the fine cap grid (node n02) with 95% intervals."""
    s = pd.read_csv(f"{RES}/e1n02_1029_0502_summary.csv")
    s = s[s.low == "zero"].copy()
    s["ci95"] = (s.speedup_hi - s.speedup_lo) / 2
    return s.rename(columns={"speedup_pct": "gain_pct"})


def fig_gemm_two_platforms():
    """GEMM speedup vs cap, GH200 (left) and MI350X (right), same series where both exist."""
    gh = pd.read_csv(f"{GH}/e1_ci.csv")
    mi = mi350_e1()
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.1), sharey=True)
    series_gh = [("fp64", 35, "FP64, 35 frac. bits", C[0], "o"), ("fp64", 27, "FP64, 27", C[1], "s"),
                 ("bf16", 3, "BF16, 3", C[3], "^"), ("bf16", 0, "BF16, 0", C[2], "D")]
    series_mi = [("fp64", 36, "FP64, 36 frac. bits", C[0], "o"), ("fp64", 27, "FP64, 27", C[1], "s"),
                 ("bf16", 3, "BF16, 3", C[3], "^"), ("bf16", 0, "BF16, 0", C[2], "D"),
                 ("fp8", 0, "FP8 E4M3, 0", C[4], "v")]
    for ax, d, series, title, xcol in ((axes[0], gh, series_gh, "NVIDIA GH200 (module cap)", "tdp"),
                                       (axes[1], mi, series_mi, "AMD MI350X (PPT cap)", "cap")):
        for dt, fr, lab, c, mk in series:
            g = d[(d.dtype == dt) & (d.frac == fr)].sort_values(xcol)
            if g.empty:
                continue
            ax.errorbar(g[xcol], g.gain_pct, yerr=g.ci95, label=lab, color=c, marker=mk, ms=3, lw=1, capsize=2)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.axvline(1000, color="#bbbbbb", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Power cap (W)")
        ax.legend(frameon=False, ncol=2, loc="upper right")
    axes[0].set_ylabel("GEMM speedup vs. native (%)")
    axes[0].set_ylim(-2, 70)
    save(fig, "mi350_gemm_caps.pdf")


def fig_model():
    """Predicted vs measured speedup: GH200 (one-cap calibration) and MI350X (fixed-clock R and sigma)."""
    d = pd.read_csv(f"{RES}/model_pred.csv")
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    sel = [("GH200", "onecap", "GH200: calibrated at the lowest cap", C[0], "o"),
           ("MI350X", "fixed+stretch", "MI350X: from fixed-clock power and clock", C[1], "s")]
    for plat, meth, lab, c, mk in sel:
        g = d[(d.platform == plat) & (d.method == meth)]
        if meth == "onecap":  # the calibration point itself is not a prediction
            g = g[g.cap != g.groupby(["dtype", "frac"]).cap.transform("min")]
        mae = (g.pred_gain - g.meas_gain).abs().mean()
        ax.scatter(g.meas_gain, g.pred_gain, s=10, color=c, marker=mk, label=f"{lab} (MAE {mae:.1f} pts)",
                   alpha=0.8, lw=0)
    lim = [0, max(d.meas_gain.max(), d.pred_gain.max()) * 1.05]
    ax.plot(lim, lim, color="#999999", lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Measured speedup (%)")
    ax.set_ylabel("Predicted speedup (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    save(fig, "mi350_model.pdf")


def fig_crossover():
    """MI350X: truncation vs GEMMul8 speedup over native FP64 as a function of k, at three caps."""
    s = pd.read_csv(f"{RES}/xover_1029_1508_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.0), sharey=True)
    series = [("fp64", 35, "Truncate, 35 frac. bits (7.8e-12)", C[0], "o", "-"),
              ("fp64", 27, "Truncate, 27 (2.0e-9)", C[0], "s", "--"),
              ("g8", 10, "GEMMul8, 10 moduli (1.0e-10)", C[1], "^", "--"),
              ("g8", 12, "GEMMul8, 12 moduli (4–8e-13)", C[1], "D", "-"),
              ("g8", 14, "GEMMul8, 14 moduli (2–7e-15)", C[1], "v", ":")]
    for ax, cap in zip(axes, (1000, 700, 500)):
        for dt, fr, lab, c, mk, ls in series:
            g = s[(s.dtype == dt) & (s.frac == fr) & (s.cap == cap)].sort_values("k")
            ax.plot(g.k, g.speedup_pct, color=c, marker=mk, ms=3, lw=1, ls=ls, label=lab)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096, 8192], ["1024", "2048", "4096", "8192"])
        ax.set_title(f"{cap} W")
        ax.set_xlabel("Inner dimension $k$ ($m=n=32768$)")
    axes[0].set_ylabel("Speedup vs. native FP64 (%)")
    axes[2].legend(frameon=False, loc="upper left", fontsize=6)
    save(fig, "mi350_crossover.pdf")


def fig_apps():
    """MI350X applications: speedup vs native under each cap (truncation operating points)."""
    rows = []
    for name, f, var, lab in [("hpl", "hpl_1029_0336", "36:36", "FP64 HPL, 36 bits"),
                              ("mxp", "mxp1028", "3", "HPL-MxP, 3 bits"),
                              ("gp", "gp_1029_0731", "t28n", "GP likelihood, 28 bits"),
                              ("mp2", "mp2_1029_1607", "t20n", "DF-MP2, 20 bits")]:
        path = f"{RES}/{'mxp_1028_2359' if name == 'mxp' else f}_summary.csv"
        s = pd.read_csv(path)
        g = s[s.variant.astype(str) == var]
        for r in g.itertuples():
            rows.append(dict(app=lab, cap=r.cap, gain=r.speedup_pct, lo=r.speedup_lo, hi=r.speedup_hi))
    d = pd.DataFrame(rows)
    caps = [1000, 800, 650, 500]
    apps = list(dict.fromkeys(d.app))
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    w = 0.8 / len(apps)
    for i, a in enumerate(apps):
        g = d[d.app == a].set_index("cap").reindex(caps)
        x = np.arange(len(caps)) + (i - (len(apps) - 1) / 2) * w
        ax.bar(x, g.gain, w, color=C[i], label=a, yerr=[g.gain - g.lo, g.hi - g.gain], capsize=1.5,
               error_kw={"lw": 0.6})
    ax.set_xticks(np.arange(len(caps)), [f"{c} W" for c in caps])
    ax.set_ylabel("Speedup vs. native (%)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    save(fig, "mi350_apps.pdf")


if __name__ == "__main__":
    fig_gemm_two_platforms()
    fig_model()
    fig_crossover()
    fig_apps()
