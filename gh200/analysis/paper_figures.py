"""Vector figures for the paper manuscript (paper/figs/*.pdf), drawn from the summarized results.

usage: paper_figures.py [outdir]   (default ../paper/figs relative to gh200/)
Every figure reads only files under results/ (and the span/range numbers quoted in the text), so the paper
figures can be regenerated after a re-analysis.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

# TrueType (Type 42) fonts in the PDFs: IEEE PDF eXpress rejects Type 3 fonts
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "paper", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.45      # IEEE single column width (in)
WIDE = 7.0      # double column
C = ["#2a6f97", "#d1495b", "#edae49", "#66a182", "#8d6cab", "#555555"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    if os.environ.get("PREVIEW"):  # PNG copies for a quick visual check
        os.makedirs(os.environ["PREVIEW"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["PREVIEW"], name.replace(".pdf", ".png")), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def fig_gemm():
    d = pd.read_csv("results/e1_ci.csv")
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    series = [("fp64", 35, "FP64, 35 bits kept", C[0], "o"), ("fp64", 27, "FP64, 27 bits kept", C[1], "s"),
              ("bf16", 3, "BF16, 3 bits kept", C[3], "^"), ("bf16", 0, "BF16, 0 bits kept", C[2], "D")]
    for dt, fr, lab, c, mk in series:
        g = d[(d.dtype == dt) & (d.frac == fr)].sort_values("tdp")
        ax.errorbar(g.tdp, g.gain_pct, yerr=g.ci95, label=lab, color=c, marker=mk, ms=3, lw=1, capsize=2)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("Module power cap (W)")
    ax.set_ylabel("GEMM speedup vs. native (%)")
    ax.set_ylim(-2, 49)  # headroom so the legend clears the 400 W BF16 point
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "gemm_caps.pdf")


def fig_hpl():
    # HPL: the tuned configuration (job 3315: N = 286720, NCCL 8 channels / 2 MiB); its native run is still
    # power-capped at 1000 W, so that point stays. HPL-MxP's native 1000 W point tripped the power brake.
    e3 = pd.read_csv("results/e3_tuned_summary.csv")
    e2 = pd.read_csv("results/e2_summary.csv")
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    g = e3[e3.variant == "36:36"].sort_values("tdp")
    ax.plot(g.tdp, g.gain_pct, "o-", color=C[0], ms=3, lw=1, label="HPL, 36 bits (8 nodes)")
    g2 = e2[e2.variant.astype(str) == "3"].sort_values("tdp")
    g2 = g2[g2.tdp < 1000]
    ax.plot(g2.tdp, g2.gain_pct, "s-", color=C[1], ms=3, lw=1, label="HPL-MxP, 3 bits (8 nodes)")
    off = {400: (5, 2), 500: (0, -9)}  # keep the labels clear of the frame and of the HPL-MxP line
    for _, r in g[g.tdp < 1000].iterrows():
        dx, dy = off.get(int(r.tdp), (0, 4))
        ax.annotate(f"+{r.dw_cap:.0f} W", (r.tdp, r.gain_pct), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "center", fontsize=6, color=C[0])
    ax.set_xlabel("Module power cap per node (W)")
    ax.set_ylabel("Speedup vs. native (%)")
    ax.set_ylim(0, 15)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "hpl_caps.pdf")


def fig_crossover():
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 2.9), sharex=True, sharey=True)
    norm = TwoSlopeNorm(vmin=np.log2(0.45), vcenter=0.0, vmax=np.log2(7.5))
    for i, (fn, name) in enumerate([("crossover_fp64emu32.csv", "cuBLAS emulation, 32 bits"),
                                    ("crossover_gemmul810.csv", "GEMMul8, 10 moduli")]):
        d = pd.read_csv(os.path.join("results", fn))
        ks = sorted(d.k.unique())
        ns = sorted(d.n.unique())
        for j, tdp in enumerate(sorted(d.tdp.unique(), reverse=True)):
            ax = axes[i, j]
            g = d[d.tdp == tdp].pivot_table(index="n", columns="k", values="ratio").reindex(index=ns, columns=ks)
            im = ax.imshow(np.log2(g.values), cmap="RdBu", norm=norm, origin="lower", aspect="auto")
            for a in range(len(ns)):
                for b in range(len(ks)):
                    v = g.values[a, b]
                    ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(np.log2(v)) > 1.6 else "black")
            ax.set_xticks(range(len(ks)), [f"{k // 1024}K" if k >= 1024 else str(k) for k in ks])
            ax.set_yticks(range(len(ns)), [str(n) for n in ns])
            if i == 0:
                ax.set_title(f"{tdp} W")
            if j == 0:
                ax.set_ylabel(f"{name}\n$n$")
            if i == 1:
                ax.set_xlabel("$k$")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    ticks = [0.5, 1, 2, 4]
    cb.set_ticks(np.log2(ticks), labels=[str(t) for t in ticks])
    cb.set_label("$T_{\\mathrm{emu}}/T_{\\mathrm{trunc36}}$ (>1: truncation faster)")
    save(fig, "crossover.pdf")


def fig_spread():
    d = pd.read_csv("results/range_sweep.csv")
    d = d[(d.smode == "elem") & (d.tdp == 1000)]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    series = [(("fp64", 36), "Truncated FP64, 36 bits", C[0], "o"), (("fp64emu", 32), "cuBLAS emulation, 32 bits", C[1], "s"),
              (("fp64emu", 40), "cuBLAS emulation, 40 bits", C[2], "^"), (("gemmul8", 10), "GEMMul8, 10 moduli", C[3], "D"),
              (("fp32", 23), "FP32", C[4], "v")]
    ss = sorted(d.spread.unique())
    for (dt, fr), lab, c, mk in series:
        g = d[(d.dtype == dt) & (d.frac == fr)].sort_values("spread")
        fin = g[np.isfinite(g.err_p50)]
        ax.plot(fin.spread, fin.err_p50, marker=mk, ms=3, lw=1, color=c, label=lab)
        for _, r in g[~np.isfinite(g.err_p50)].iterrows():
            ax.annotate("overflow", (r.spread, 2e-6), color=c, fontsize=6, ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_xscale("symlog", linthresh=16)
    ax.set_xticks(ss, [str(int(s)) for s in ss])
    ax.set_xlabel("Exponent spread within rows, $S$ (binades)")
    ax.set_ylabel("Median elementwise rel. error")
    ax.set_ylim(1e-15, 1e-4)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    save(fig, "spread_error.pdf")


def fig_map():
    # coordinates and outcomes come from results/range_map_points.csv (sources listed there)
    off = {"HPL (uniform)": (0, 7), "DF-MP2 tensor": (4, -9), "QE Au surface": (-4, 4), "QE Si": (-4, 4),
           "Gaussian-basis DFT": (-4, -10), "GP geostatistics": (-4, 4)}
    t = pd.read_csv(os.path.join(RES, "range_map_points.csv"))
    pts = [(r.label_en, r.global_span, r.frac_below32, bool(r.emu_ok), off[r.label_en]) for r in t.itertuples()]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for x, lab in [(18, "FP8 E4M3"), (40, "FP16"), (254, "FP32/BF16")]:
        ax.axvline(x, color="#999999", lw=0.6, ls="--")
        ax.text(x * 1.05, 7e-7, f"{lab}\nrange", fontsize=6, color="#666666", va="bottom")
    for lab, g, r, ok, off in pts:
        ax.scatter([g], [r], s=22, color=C[0] if ok else C[1], zorder=3)
        ax.annotate(lab, (g, r), textcoords="offset points", xytext=off, fontsize=6.5,
                    ha="center" if off[0] == 0 else "right" if off[0] < 0 else "left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(8, 1024)
    ax.set_ylim(5e-7, 2)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024], ["8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.set_xlabel("Global exponent span (binades, central 99.9%)")
    ax.set_ylabel("Fraction of elements\n$>2^{-32}$ below row maximum")
    ax.scatter([], [], color=C[0], s=18, label="fixed 32–36-bit emulation usable")
    ax.scatter([], [], color=C[1], s=18, label="fixed-bit emulation silently wrong")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
    save(fig, "range_map.pdf")


if __name__ == "__main__":
    for f in (fig_gemm, fig_hpl, fig_crossover, fig_spread, fig_map):
        f()
