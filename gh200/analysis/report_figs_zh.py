#!/usr/bin/env python3
"""Figures 5.1-5.13 of the Traditional-Chinese project report (專題成果報告, LaTeX, A4, text width 15.5 cm).

usage: report_figs_zh.py <font dir> [outdir]
  font dir  directory holding NotoSansTC.ttf (variable TrueType, preferred) or NotoSansCJKtc-Regular.otf
  outdir    default: <repo>/paper/zh_report/figs
  CHECK=1   also report text that leaves the figure or overlaps other text

Every measured number (values, labels, legend errors) is read from gh200/results/ or gh200/logs/ (the source is
named in each function); only format constants (the exponent ranges in 5.13) are written here.
Fonts: NotoSansTC.ttf is a variable font whose default instance is Thin, so the Regular (wght 400) instance is cut
out with fontTools into a temporary directory and embedded as TrueType (Type 42); without fontTools the CFF-based
NotoSansCJKtc-Regular.otf is used and embedded as Type 3. The PDFs carry no creation date, so reruns are
byte-identical.

Style: no titles (the LaTeX caption carries them); grey for context, one blue accent for the subject of each
figure, orange only for a genuinely different category; 8-9 pt text at the printed size (15.5 or 7.5 cm wide).
Bit counts follow the report: kept fraction bits (FP64 native = 52), except HPL, which counts the hidden bit as
rocHPL does (native = 53; "36 位" = 35 fraction bits).
"""
import io
import json
import os
import re
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOGS = os.path.join(HERE, "..", "logs")

CM = 1 / 2.54
FULL, HALF = 15.5, 7.5  # printed widths (cm)

# ink and chrome
INK, INK2, MUTED = "#1a1a19", "#52514e", "#8a8984"
GRID, SPINE = "#e3e2dc", "#8a8984"
# one accent hue (blue ramp), greys for context, orange only for a different category
BLUE_D, BLUE, BLUE_L, BLUE_XL = "#1c5cab", "#2a78d6", "#86b6ef", "#cde2fb"
GREY_D, GREY, GREY_L = "#5c5b57", "#8c8b86", "#bdbcb5"
ORANGE, ORANGE_M, ORANGE_L = "#e0602b", "#f29c74", "#fbd9c8"

# wording shared by several figures
REL_GAIN = "相對同一上限下原生的加速（%）"
REL_SPEED = "相對同一上限下原生的速度（倍）"
CAP = "模組功耗上限（W）"
CAP_NODE = "每節點模組功耗上限（W）"


# ----------------------------------------------------------------------------------------------- setup
def setup_fonts(font_dir, tmp):
    """Register the CJK sans font with matplotlib; return its family name."""
    ttf = os.path.join(font_dir, "NotoSansTC.ttf")
    otf = os.path.join(font_dir, "NotoSansCJKtc-Regular.otf")
    path, fonttype = None, 3
    if os.path.exists(ttf):
        try:
            from fontTools.ttLib import TTFont
            from fontTools.varLib import instancer
            print("從可變字型切出 Regular（wght 400）……")
            src = TTFont(ttf, recalcTimestamp=False)  # keep head.modified so the PDFs are byte-reproducible
            font = instancer.instantiateVariableFont(src, {"wght": 400}, updateFontNames=True)
            font.recalcTimestamp = False
            path = os.path.join(tmp, "NotoSansTC-Regular.ttf")
            font.save(path)
            fonttype = 42
        except ImportError:
            path = None
    if path is None:
        if not os.path.exists(otf):
            sys.exit(f"找不到字型：{ttf} 或 {otf}")
        path = otf
    font_manager.fontManager.addfont(path)
    name = font_manager.FontProperties(fname=path).get_name()
    print(f"字型：{name}（{os.path.basename(path)}，PDF Type {fonttype}）")
    plt.rcParams.update({"pdf.fonttype": fonttype, "ps.fonttype": fonttype})
    return name


def setup_style(family):
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.title_fontsize": 8, "axes.titlesize": 8.5,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": family, "mathtext.bf": family,
        "mathtext.sf": family, "mathtext.tt": family, "mathtext.cal": family, "mathtext.default": "regular",
        "axes.unicode_minus": True,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SPINE, "axes.linewidth": 0.6,
        "xtick.color": SPINE, "ytick.color": SPINE, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "lines.linewidth": 1.3, "lines.markersize": 4, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.4, "legend.labelspacing": 0.35,
        "errorbar.capsize": 1.8, "hatch.linewidth": 0.5,
        "pdf.compression": 9, "savefig.dpi": 300, "figure.dpi": 100,
    })


def res(name):
    return os.path.join(RES, name)


def read_table(name):
    """The *_summary.txt files are whitespace-separated tables with one header line."""
    return pd.read_csv(io.StringIO(open(res(name)).read()), sep=r"\s+")


def fig_axes(w, h, left, right, bottom, top):
    """A figure of w x h cm with one axes placed by its margins (cm)."""
    fig = plt.figure(figsize=(w * CM, h * CM))
    ax = fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h])
    return fig, ax


def pct(v, nd=1, plus=True):
    s = f"{v:.{nd}f}"
    return ("+" if plus and v > 0 else "") + s + "%"


def mant_exp(v, nd=1):
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    if round(m, nd) >= 10:
        m, e = m / 10, e + 1
    return m, e


def sci(v, nd=1):
    """4.8e-6 -> mathtext 4.8×10^{-6}."""
    m, e = mant_exp(v, nd)
    return rf"${m:.{nd}f}{{\times}}10^{{{e}}}$"


def sci_range(vals, nd=1):
    """Measured errors as '4.8×10^-6', or '(1.0–8.2)×10^-10' when they differ."""
    (ml, el), (mh, eh) = mant_exp(min(vals), nd), mant_exp(max(vals), nd)
    if (round(ml, nd), el) == (round(mh, nd), eh):
        return sci(min(vals), nd)
    if el == eh:
        return rf"$({ml:.{nd}f}$–${mh:.{nd}f}){{\times}}10^{{{el}}}$"
    return sci(min(vals), nd) + "–" + sci(max(vals), nd)


def native_tick(ax, ticks, nd):
    """Label the y = 1 reference line on the axis itself ('原生 1.0')."""
    ax.set_yticks(ticks, [("原生 " if abs(t - 1) < 1e-9 else "") + f"{t:.{nd}f}" for t in ticks])


def left_legend(leg):
    leg._legend_box.align = "left"  # legend title flush left
    return leg


OUT = None


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, metadata={"Creator": "gh200/analysis/report_figs_zh.py", "CreationDate": None})
    if os.environ.get("CHECK"):
        check_layout(fig, name)
    plt.close(fig)
    print(f"已寫出 {os.path.relpath(path)}")


def check_layout(fig, name):
    """Warn about text outside the figure and text boxes that overlap each other."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(r)
        if b.width == 0:
            continue
        if b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5:
            print(f"  [{name}] 文字超出圖框：{t.get_text()!r}")
        boxes.append((t.get_text(), b))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ov = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if ov[0] > 0.5 and ov[1] > 0.5:
                print(f"  [{name}] 文字重疊：{boxes[i][0]!r} ↔ {boxes[j][0]!r}")


# ----------------------------------------------------------------------------------------------- 5.1
def fig_mechanism():
    """5.1 Savings at 1000 W, 0 bits kept, when the dropped bits are filled with 0 / 1 / a fixed pattern.
    m1_toggle_summary.csv (FP64 tensor-core GEMM, jobs 1752-1753), m4_fp64core_summary.txt (FMA loop, job 1815),
    m4_stream_summary.txt (copy, job 1858). Native = 52 bits kept. (The 500 W repeat, m1_toggle_500_raw.csv, is
    only quoted in the text.)"""
    fills = [("zero", "填 0（截斷）", GREY_L), ("ones", "填 1（不翻轉、非零）", BLUE),
             ("const", "固定樣式（不翻轉、非零）", BLUE_L)]
    m1 = pd.read_csv(res("m1_toggle_summary.csv"))
    tc = m1[(m1.dtype == "fp64") & (m1.tdp == 1000)]
    levels = [("FP64 tensor core\n矩陣乘法", tc[tc.frac == 52]["mod"].iloc[0],
               {lo: tc[(tc.frac == 0) & (tc.low == lo)]["mod"].iloc[0] for lo, _, _ in fills})]
    for fn, dt, lab in (("m4_fp64core_summary.txt", "dfma", "一般 FP64 運算單元\n乘加迴圈"),
                        ("m4_stream_summary.txt", "copy", "記憶體複製")):
        t = read_table(fn)
        g = t[(t.dtype == dt) & (t.tdp == 1000)]
        levels.append((lab, g[g.frac == 52].W.iloc[0], {lo: g[(g.frac == 0) & (g.low == lo)].W.iloc[0]
                                                         for lo, _, _ in fills}))
    fig, ax = fig_axes(FULL, 6.4, 3.0, 0.3, 1.0, 0.75)
    bh, gap, pitch = 0.25, 0.035, 1.12
    for gi, (lab, nat, w) in enumerate(levels):
        y0 = -gi * pitch
        full = nat - w["zero"]
        for fi, (lo, _, c) in enumerate(fills):
            y = y0 - fi * (bh + gap)
            saved = nat - w[lo]
            ax.barh(y, saved, height=bh, color=c, edgecolor="none")
            txt = f"{saved:.0f} W" if lo == "zero" else f"{saved:.0f} W（{saved / full * 100:.0f}%）"
            ax.text(saved + 4, y, txt, va="center", ha="left", fontsize=8, color=INK2 if lo == "zero" else INK)
        # category label block (name + native power in parentheses), centred on the group
        ymid = y0 - (bh + gap)
        n = lab.count("\n") + 1
        lh_main, lh_sub = 8.5 * 1.2, 8 * 1.2
        top = (n * lh_main + lh_sub) / 2
        ax.annotate(lab, (0, ymid), xycoords=("axes fraction", "data"), textcoords="offset points",
                    xytext=(-6, top), ha="right", va="top", fontsize=8.5, color=INK, linespacing=1.2)
        ax.annotate(f"（原生 {nat:.0f} W）", (0, ymid), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(-6, top - n * lh_main - 1), ha="right", va="top",
                    fontsize=8, color=INK2)
    ax.set_yticks([])
    ax.set_ylim(-(len(levels) - 1) * pitch - 2 * (bh + gap) - 0.3, 0.3)
    ax.set_xlim(0, 380)
    ax.set_xticks([0, 100, 200, 300])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("保留 0 位時比原生少用的模組功耗（W）")
    handles = [Patch(color=c, label=lab) for _, lab, c in fills]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2,
              handlelength=1.2, handleheight=0.9)
    save(fig, "fig5_01_mechanism.pdf")


# ----------------------------------------------------------------------------------------------- 5.2
def fig_gemm_caps():
    """5.2 GEMM microbenchmark speedup vs power cap with 95% CIs. e1_ci.csv (jobs 1667-1669)."""
    d = pd.read_csv(res("e1_ci.csv"))
    # (dtype, kept bits, label, colour, marker, x dodge in W so that close points and error bars stay visible)
    series = [("fp64", 35, "FP64 保留 35 位", BLUE, "o", 0), ("fp64", 27, "FP64 保留 27 位", BLUE_L, "s", -7),
              ("bf16", 0, "BF16 保留 0 位", ORANGE, "^", 7)]
    fig, ax = fig_axes(FULL, 6.6, 1.35, 1.25, 1.0, 0.25)
    for dt, fr, lab, c, mk, dx in series:
        g = d[(d.dtype == dt) & (d.frac == fr)].sort_values("tdp")
        ax.errorbar(g.tdp + dx, g.gain_pct, yerr=[g.gain_pct - g.lo, g.hi - g.gain_pct], color=c, marker=mk, ms=4,
                    lw=1.3, elinewidth=0.8, capthick=0.8, label=lab, zorder=3)
        r = g[g.tdp == 400].iloc[0]
        ax.annotate(pct(r.gain_pct), (400 + dx, r.hi), textcoords="offset points", xytext=(0, 3), ha="center",
                    va="bottom", fontsize=8, color=INK)
        r = g[g.tdp == 1000].iloc[0]
        ax.annotate(pct(r.gain_pct), (1000 + dx, r.gain_pct), textcoords="offset points", xytext=(6, 0),
                    ha="left", va="center", fontsize=8, color=INK)
    ax.set_xlim(270, 1030)
    ax.set_xticks(range(300, 1001, 100))
    ax.set_ylim(0, 40)
    ax.set_yticks(range(0, 41, 10))
    ax.set_xlabel(CAP)
    ax.set_ylabel(REL_GAIN)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig5_02_gemm_caps.pdf")


# ----------------------------------------------------------------------------------------------- 5.3
def fig_hpl():
    """5.3 HPL (tuned, 8 nodes, N = 286720): TFLOP/s vs cap, gains of 36 bits, ΔW at 700 W.
    e3_tuned_summary.csv (job 3315). rocHPL counts the hidden bit: 36 bits = 35 fraction bits."""
    e3 = pd.read_csv(res("e3_tuned_summary.csv"))
    nat = e3[e3.variant == "53:53"].sort_values("tdp")
    b36 = e3[e3.variant == "36:36"].sort_values("tdp")
    b28 = e3[e3.variant == "28:28"].sort_values("tdp")
    fig, ax = fig_axes(FULL, 7.0, 1.85, 0.3, 1.6, 0.2)
    ax.plot(b28.tdp, b28.perf, color=GREY_L, lw=1.0, ls=(0, (4, 2.5)), marker="o", ms=2.8, zorder=2,
            label="保留 28 位（失效邊界）")
    ax.plot(nat.tdp, nat.perf, color=GREY_D, lw=1.3, marker="o", ms=3.6, zorder=3, label="原生 FP64")
    ax.plot(b36.tdp, b36.perf, color=BLUE, lw=1.6, marker="o", ms=4.2, zorder=4,
            label="保留 36 位（含隱藏位；安全設定）")
    # ΔW at 700 W: the cap at which the native curve reaches the 36-bit performance (linear interpolation)
    p36 = b36[b36.tdp == 700].perf.iloc[0]
    cap_eq = float(np.interp(p36, nat.perf, nat.tdp))
    dw = cap_eq - 700
    assert abs(dw - b36[b36.tdp == 700].dw_cap.iloc[0]) < 0.5, "ΔW differs from e3_tuned_summary.csv"
    ax.annotate("", xy=(cap_eq, p36), xytext=(700, p36), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=INK, lw=0.8,
                                shrinkA=3, shrinkB=0))
    ax.plot([cap_eq, cap_eq], [p36, p36 - 33], color=INK2, lw=0.6, ls=(0, (1.5, 1.5)), zorder=5)
    ax.text(cap_eq + 6, p36 - 33, f"相當於多給 {dw:.0f} W 上限", ha="left", va="center", fontsize=8, color=INK)
    # speedups of 36 bits, as a row right under the cap labels
    for _, r in b36.iterrows():
        g = (r.perf / nat[nat.tdp == r.tdp].perf.iloc[0] - 1) * 100
        ax.annotate(pct(g), (r.tdp, 0), xycoords=("data", "axes fraction"), textcoords="offset points",
                    xytext=(0, -18), ha="center", va="top", fontsize=8, color=INK)
    ax.annotate("36 位加速", (0, 0), xycoords="axes fraction", textcoords="offset points",
                xytext=(-4, -18), ha="right", va="top", fontsize=8, color=INK2)
    ax.set_xlim(370, 1030)
    ax.set_xticks(sorted(b36.tdp))  # only the measured caps (900 W was not run)
    ax.set_ylim(200, 425)
    ax.set_yticks(range(200, 401, 50))
    ax.set_xlabel(CAP_NODE, labelpad=20)
    ax.set_ylabel("HPL 效能（TFLOP/s）")
    h, lab = ax.get_legend_handles_labels()
    ax.legend([h[2], h[1], h[0]], [lab[2], lab[1], lab[0]], loc="upper left")
    save(fig, "fig5_03_hpl.pdf")


# ----------------------------------------------------------------------------------------------- 5.4
def fig_hplmxp():
    """5.4 HPL-MxP (8 nodes, N = 491520) speedup per cap, 3 and 0 bits kept. e2_summary.csv (job 1671).
    The native 1000 W run tripped the chassis power brake, so 1000 W is shown as excluded."""
    e2 = pd.read_csv(res("e2_summary.csv"))
    e2["variant"] = e2.variant.astype(str)
    caps = [400, 500, 600, 700, 800]
    series = [("3", "保留 3 位（安全設定）", BLUE, INK), ("0", "保留 0 位（失效邊界）", GREY_L, INK2)]
    fig, ax = fig_axes(FULL, 6.2, 1.35, 0.3, 1.0, 0.2)
    bw = 0.34
    for j, (v, lab, c, tc) in enumerate(series):
        g = e2[e2.variant == v].set_index("tdp")
        xs = [i + (j - 0.5) * (bw + 0.04) for i in range(len(caps))]
        vals = [g.gain_pct[cap] for cap in caps]
        ax.bar(xs, vals, width=bw, color=c, label=lab, zorder=3)
        for x, val in zip(xs, vals):
            ax.text(x, val + 0.5, pct(val), ha="center", va="bottom", fontsize=8, color=tc)
    n = len(caps)
    ax.text(n, 1.0, "原生在 1000 W 觸發\n機箱硬體功耗煞車，\n故不列入比較", ha="center", va="bottom", fontsize=8,
            color=INK2, linespacing=1.3)
    ax.set_xticks(range(n + 1), [str(c) for c in caps] + ["1000"])
    ax.set_xlim(-0.6, n + 0.6)
    ax.set_ylim(0, 30)
    ax.set_yticks(range(0, 31, 10))
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel(CAP_NODE)
    ax.set_ylabel(REL_GAIN)
    ax.legend(loc="upper right", bbox_to_anchor=(0.86, 1.0))
    save(fig, "fig5_04_hplmxp.pdf")


# ----------------------------------------------------------------------------------------------- 5.5
def qe_energies(path):
    """Final '!' total energy of each variant in a sci_sweep log (None when the run failed)."""
    out, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"=== RUN tdp=\d+ variant=(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = None
            continue
        m = re.match(r"\s*!\s+total energy\s+=\s+([-\d.]+) Ry", line)
        if m and cur:
            out[cur] = float(m.group(1))
    return out


def fig_qe_rounding():
    """5.5 QE gold surface (ausurf112), all FP64 GEMMs intercepted at 1000 W: |E - E_native| vs kept bits,
    truncation toward zero vs round-to-nearest. logs/sci/sci-ausurf_1729.out and sci-ausurf2_1738.out; reference =
    the unmodified 'stats' run of job 1729. 44:trunc (|ΔE| = 0 at the printed digits) and 12:trunc (failed) lie
    outside the plotted range; 24:trunc was not run. The brackets give the error ratio at equal kept bits."""
    e = qe_energies(os.path.join(LOGS, "sci", "sci-ausurf_1729.out"))
    e.update(qe_energies(os.path.join(LOGS, "sci", "sci-ausurf2_1738.out")))
    ref = e["stats"]
    bits = [36, 32, 28, 24, 20, 16]
    y_div = 1e-3
    fig, ax = fig_axes(HALF, 6.6, 1.55, 0.3, 1.0, 0.75)
    err = {}
    for mode, lab, c, mk in (("nearest", "最近捨入", BLUE, "o"), ("trunc", "往零截斷", GREY_D, "s")):
        pts = [(bits.index(b), e[f"{b}:{mode}"]) for b in bits if f"{b}:{mode}" in e]
        ok = [(i, abs(v - ref)) for i, v in pts if v is not None]
        div = [i for i, v in pts if v is None]
        first = min(i for i, _ in pts)
        missing = [bits[i] for i in range(first, max(i for i, _ in pts)) if f"{bits[i]}:{mode}" not in e]
        if missing:
            lab += "（" + "、".join(str(b) for b in missing) + " 位未測）"
        err[mode] = dict(ok)
        ax.plot([p[0] for p in ok], [p[1] for p in ok], color=c, marker=mk, ms=4, lw=1.3, label=lab, zorder=3)
        if div:
            i = min(div)
            ax.plot([ok[-1][0], i], [ok[-1][1], y_div], color=c, lw=0.9, ls=(0, (2.5, 2)), zorder=2)
            ax.plot([i], [y_div], marker="X", ms=6.5, color=c, mec="white", mew=0.6, ls="none", zorder=4)
    # error ratio truncation / nearest wherever both converged
    for i in sorted(set(err["nearest"]) & set(err["trunc"])):
        lo, hi = err["nearest"][i], err["trunc"][i]
        ax.plot([i, i], [lo, hi], color=INK2, lw=0.7, zorder=2)
        ax.text(i, np.sqrt(lo * hi), f"{hi / lo:.0f} 倍", ha="center", va="center", fontsize=8, color=INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_yscale("log")
    ax.set_ylim(10 ** -8.6, 10 ** -2.7)
    ax.axhspan(10 ** -3.7, 10 ** -2.7, color="#f4f3ef", lw=0, zorder=0)
    ax.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, y_div],
                  [r"$10^{-8}$", r"$10^{-7}$", r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", "發散"])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xticks(range(len(bits)), [str(b) for b in bits])
    ax.set_xlim(-0.5, len(bits) - 0.5)
    ax.set_xlabel("保留的尾數位數（越往右越少）")
    ax.set_ylabel(r"$|E-E_{\mathrm{原生}}|$（Ry）")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.2, 1.0), ncol=2, borderaxespad=0.2, columnspacing=1.0)
    save(fig, "fig5_05_qe_rounding.pdf")


# ----------------------------------------------------------------------------------------------- 5.6, 5.7
EMU_SERIES = [(("fp64", 36), "截斷 36 位", BLUE, "o", "-"),
              (("fp64emu", 32), "cuBLAS 模擬 32 位", GREY_D, "s", "-"),
              (("gemmul8", 10), "GEMMul8 10 模數", GREY, "D", (0, (3.5, 2)))]
SP_SERIES = [(("tf32x3", 23), "3xTF32", GREY_D, "^", (0, (1, 1.5))),
             (("tfp32", 23), "FP32", GREY, "v", (0, (1, 1.5)))]


def row(t, dtype, frac, tdp, job=None):
    g = t[(t.dtype == dtype) & (t.frac == frac) & (t.tdp == tdp)]
    if job is not None:
        g = g[g.job == job]
    assert len(g) == 1, (dtype, frac, tdp, job)
    return g.iloc[0]


def tflops(t, dtype, frac, tdp, job=None):
    return row(t, dtype, frac, tdp, job).tflops


def err_labels(series, cells, first_prefix="誤差 "):
    """Legend labels with the measured relative error (column err), e.g. '截斷 36 位（誤差 1.0×10^-11）';
    cells = [(table, tdp, job), ...] over which the errors are collected."""
    out = []
    for i, ((dt, fr), name, *_) in enumerate(series):
        errs = [row(t, dt, fr, tdp, job).err for t, tdp, job in cells]
        out.append(f"{name}（{first_prefix if i == 0 else ''}{sci_range(errs)}）")
    return out


def fig_emu_caps():
    """5.6 Square GEMM 32768x32768x2048: speed relative to native FP64 at each cap. Left: methods at matched
    accuracy (truncation, cuBLAS INT8 emulation, GEMMul8); right: single-precision alternatives (3xTF32, FP32),
    whose errors are 4-5 orders of magnitude larger. Shared y scale. m3_ozaki2_tf32x3_summary.txt (job 1894)."""
    t = read_table("m3_ozaki2_tf32x3_summary.txt")
    caps = [450, 575, 1000]
    w, h = FULL, 8.0
    left, right, bottom, top, gap = 1.75, 0.3, 1.0, 1.95, 0.6
    wl = (w - left - right - gap) * 0.6
    wr = w - left - right - gap - wl
    fig = plt.figure(figsize=(w * CM, h * CM))
    axl = fig.add_axes([left / w, bottom / h, wl / w, (h - bottom - top) / h])
    axr = fig.add_axes([(left + wl + gap) / w, bottom / h, wr / w, (h - bottom - top) / h], sharey=axl)
    cells = [(t, cap, None) for cap in caps]
    for ax, series, labels, title in (
            (axl, EMU_SERIES, err_labels(EMU_SERIES, cells), "誤差對齊的做法"),
            (axr, SP_SERIES, err_labels(SP_SERIES, cells), "單精度做法（誤差未對齊）")):
        ax.axhline(1.0, color=GREY_D, lw=0.7, zorder=2)
        for ((dt, fr), _, c, mk, ls), lab in zip(series, labels):
            ys = [tflops(t, dt, fr, cap) / tflops(t, "fp64", 52, cap) for cap in caps]
            ax.plot(caps, ys, color=c, marker=mk, ms=4 if mk != "D" else 3.6, lw=1.3, ls=ls, label=lab, zorder=3)
            for cap, y in zip(caps, ys):
                if dt == "gemmul8" and cap == caps[0]:  # right of the diamond, clear of the cuBLAS label below
                    xy, ha, va = (6, 0), "left", "center"
                elif dt in ("fp64emu", "tfp32"):
                    xy, ha, va = (0, -5), "center", "top"
                else:
                    xy, ha, va = (0, 5), "center", "bottom"
                ax.annotate(f"{y:.2f}", (cap, y), textcoords="offset points", xytext=xy, ha=ha, va=va, fontsize=8,
                            color=INK)
        ax.set_xlim(400, 1050)
        ax.set_xticks(caps)
        left_legend(ax.legend(title=title, loc="lower left", bbox_to_anchor=(-0.02, 1.01), borderaxespad=0.1))
    axl.set_ylim(0.72, 1.76)
    native_tick(axl, [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], 1)
    axr.tick_params(axis="y", labelleft=False)
    axl.set_ylabel(REL_SPEED)
    fig.text((left + (w - left - right) / 2) / w, 0.12 / h, CAP, ha="center", va="bottom", fontsize=9, color=INK)
    save(fig, "fig5_06_emu_caps.pdf")


def fig_shapes():
    """5.7 Speed relative to native FP64 at 1000 W for the square shape (reference, same data as 5.6) and the two
    valinomycin shapes. m3_ozaki2_tf32x3_summary.txt (job 1894, square), m3_realshape_summary.txt (job 1896
    tall-skinny, 1897 long inner dimension)."""
    sq, rs = read_table("m3_ozaki2_tf32x3_summary.txt"), read_table("m3_realshape_summary.txt")
    shapes = [("方陣（合成資料）", "32768 × 32768 × 2048", sq, None),
              ("瘦長（纈胺黴素）", "757248 × 300 × 2958", rs, 1896),
              ("長內積（纈胺黴素）", "2958 × 2958 × 76800", rs, 1897)]
    fig, ax = fig_axes(FULL, 6.2, 1.75, 0.3, 1.45, 0.2)
    bw = 0.25
    labels = err_labels(EMU_SERIES, [(t, 1000, job) for _, _, t, job in shapes])
    for j, ((dt, fr), _, c, _, _) in enumerate(EMU_SERIES):
        xs, ys = [], []
        for i, (_, _, t, job) in enumerate(shapes):
            xs.append(i + (j - 1) * (bw + 0.03))
            ys.append(tflops(t, dt, fr, 1000, job) / tflops(t, "fp64", 52, 1000, job))
        ax.bar(xs, ys, width=bw, color=c, label=labels[j], zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(1.0, color=GREY_D, lw=0.7, zorder=4)
    ax.set_xticks(range(len(shapes)), [f"{a}\n{b}" for a, b, _, _ in shapes])
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.set_xlim(-0.55, len(shapes) - 0.45)
    ax.set_ylim(0, 2.5)
    native_tick(ax, [0, 0.5, 1.0, 1.5, 2.0], 1)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("矩陣形狀 m × n × k")
    ax.set_ylabel(REL_SPEED)
    left_legend(ax.legend(title="功耗上限 1000 W", loc="upper left", bbox_to_anchor=(0.0, 1.0)))
    save(fig, "fig5_07_shapes.pdf")


# ----------------------------------------------------------------------------------------------- 5.8
def fig_crossover():
    """5.8 Crossover sweep (m = 8192, two rounds): emulation time / truncation(36 bits) time per (n, k, cap).
    crossover_fp64emu32.csv and crossover_gemmul810.csv (jobs 2005-2008). Class edges are reciprocal around 1
    (0.5|2, 0.8|1.25), so equal shades mean equal factors in either direction."""
    edges = [0, 0.5, 0.8, 1.0, 1.25, 2, 4, np.inf]
    colors = [ORANGE, ORANGE_M, ORANGE_L, BLUE_XL, BLUE_L, BLUE, BLUE_D]
    white_text = {BLUE, BLUE_D}
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(colors) + 1)), len(colors))
    methods = ["crossover_fp64emu32.csv", "crossover_gemmul810.csv"]
    caps = [450, 575, 1000]
    w, h = FULL, 9.6
    fig = plt.figure(figsize=(w * CM, h * CM))
    left, right, gap = 1.75, 0.25, 0.3
    pw = (w - left - right - 2 * gap) / 3
    ph = 2.55
    rows_top = [h - 1.05, h - 1.05 - ph - 1.25]
    for mi, fn in enumerate(methods):
        d = pd.read_csv(res(fn))
        mlab = d.method.iloc[0]  # e.g. 'cuBLAS 模擬 32 位'
        ks, ns = sorted(d.k.unique()), sorted(d.n.unique())
        top = rows_top[mi]
        fig.text(left / w, (top + 0.62) / h, mlab, ha="left", va="center", fontsize=9, color=INK)
        for ci, cap in enumerate(caps):
            ax = fig.add_axes([(left + ci * (pw + gap)) / w, (top - ph) / h, pw / w, ph / h])
            g = d[d.tdp == cap].pivot_table(index="n", columns="k", values="ratio").reindex(index=ns, columns=ks)
            cls = np.digitize(g.values, edges[1:-1])
            ax.imshow(cls, cmap=cmap, norm=norm, origin="lower", aspect="auto", interpolation="nearest")
            for a in range(len(ns)):
                for b in range(len(ks)):
                    v = g.values[a, b]
                    ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if colors[cls[a, b]] in white_text else INK)
            ax.set_xticks(np.arange(len(ks) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(ns) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", lw=1.2)
            ax.grid(which="major", visible=False)
            ax.tick_params(which="both", length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks(range(len(ks)), [f"{k // 1024}K" if k >= 1024 else str(k) for k in ks])
            ax.set_yticks(range(len(ns)), [str(n) for n in ns])
            if ci > 0:
                ax.set_yticklabels([])
            ax.set_title(f"{cap} W", fontsize=8.5, color=INK, pad=3)
            if mi == 1:
                ax.set_xlabel("內積長度 k", fontsize=8.5, labelpad=2)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel("輸出寬度 n", fontsize=8.5)
    # discrete colour key
    cax = fig.add_axes([left / w, 0.55 / h, 7.0 / w, 0.3 / h])
    cax.imshow(np.arange(len(colors))[None, :], cmap=cmap, norm=norm, aspect="auto", extent=(0, len(colors), 0, 1))
    cax.set_xticks(range(1, len(colors)), [f"{e:g}" for e in edges[1:-1]])
    cax.set_yticks([])
    cax.tick_params(length=2, pad=1.5)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.grid(False)
    cax.set_xlim(0, len(colors))
    fig.text((left + 7.25) / w, 0.72 / h, "模擬時間 ÷ 截斷 36 位的時間（m = 8192）", ha="left", va="center",
             fontsize=8.5, color=INK)
    fig.text((left + 7.25) / w, 0.28 / h, "小於 1：模擬較快；大於 1：截斷較快", ha="left", va="center",
             fontsize=8, color=INK2)
    save(fig, "fig5_08_crossover.pdf")


# ----------------------------------------------------------------------------------------------- 5.9
def fig_nongemm():
    """5.9 Module power saved at 1000 W (clock 1980 MHz, performance unchanged) by kernels that are not GEMMs.
    m4_fp64core_summary.txt (FMA loop, N-body with only the inputs truncated; job 1815), m4_stream_summary.txt
    (copy, triad; job 1858), m4_cg_summary.txt (CG; job 1860)."""
    core, stream, cg = (read_table(f) for f in ("m4_fp64core_summary.txt", "m4_stream_summary.txt",
                                                "m4_cg_summary.txt"))
    kernels = [(core, "dfma", "乘加迴圈"), (stream, "copy", "記憶體複製"), (stream, "triad", "triad（讀二寫一）"),
               (core, "nbody", "N 體（僅截斷輸入）"), (cg, "cg", "CG 稀疏求解")]
    rows = []
    for t, dt, lab in kernels:
        g = t[(t.dtype == dt) & (t.tdp == 1000) & (t.low == "zero")]
        if dt == "nbody":
            g = g[g.inner == 52]
        rows.append((lab, g[g.frac == 52].W.iloc[0], {b: -g[g.frac == b].dW.iloc[0] for b in (36, 20)}))
    fig, ax = fig_axes(FULL, 6.0, 1.35, 0.3, 1.0, 0.2)
    bw = 0.34
    for j, (b, c) in enumerate(((36, BLUE), (20, BLUE_L))):  # 36 bits, the practical setting, is the strong blue
        xs = [i + (j - 0.5) * (bw + 0.04) for i in range(len(rows))]
        vals = [max(r[2][b], 0) for r in rows]
        ax.bar(xs, vals, width=bw, color=c, label=f"保留 {b} 位", zorder=3)
        for x, (lab, nat, s) in zip(xs, rows):
            ax.text(x, max(s[b], 0) + 2, f"{s[b]:.0f}" if round(s[b]) != 0 else "0", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(rows)), [f"{lab}\n（原生 {nat:.0f} W）" for lab, nat, _ in rows])
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 150)
    ax.set_yticks(range(0, 151, 50))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("比原生少用的模組功耗（W）")
    ax.legend(loc="upper right")
    save(fig, "fig5_09_nongemm.pdf")


# ----------------------------------------------------------------------------------------------- 5.10
LORA_RUNS = [  # (file in logs/lora, truncated?, seed)
    ("result_bits3_seed1234_p575.txt", False, 1234), ("result_bits3_seed1234_p800.txt", False, 1234),
    ("result_bits3_seed1234_p800_w1-8.txt", False, 1234), ("result_bits3_seed5678_p800_w2-9.txt", False, 5678),
    ("result_bits2_seed1234_p575.txt", True, 1234), ("result_bits2-trunc_seed1234_p800.txt", True, 1234),
    ("result_bits2-trunc_seed1234_p800_w1-8.txt", True, 1234),
    ("result_bits2-trunc_seed5678_p800_w2-9.txt", True, 5678)]


def lora_curve(fn):
    pts = []
    for line in open(os.path.join(LOGS, "lora", fn), errors="replace"):
        if ":::MLLOG" not in line or '"eval_accuracy"' not in line:
            continue
        r = json.loads(line.split(":::MLLOG", 1)[1])
        pts.append((r["metadata"]["samples_count"], r["value"]))
    return sorted(pts)


def fig_lora():
    """5.10 Llama 2 70B LoRA (8 nodes): validation loss vs training samples, native FP8 vs truncated to 2 bits,
    two seeds. logs/lora/result_*.txt (MLPerf eval_accuracy = validation loss; target 0.925)."""
    fig, ax = fig_axes(FULL, 6.6, 1.45, 0.3, 1.0, 0.2)
    curves = {fn: lora_curve(fn) for fn, _, _ in LORA_RUNS}
    dashed = (0, (4, 2))
    for fn, trunc, seed in LORA_RUNS:
        p = curves[fn]
        ax.plot([q[0] for q in p], [q[1] for q in p], color=BLUE if trunc else GREY, lw=1.2,
                ls="-" if seed == 1234 else dashed, marker="o", ms=2.6, zorder=3 if trunc else 2)
    ax.axhline(0.925, color=INK2, lw=0.7, zorder=1)
    ax.text(1560, 0.9253, "達標門檻 0.925", ha="left", va="bottom", fontsize=8, color=INK2)
    kinds = ax.legend(handles=[Line2D([], [], color=GREY, lw=1.6, label="原生 FP8（4 次）"),
                               Line2D([], [], color=BLUE, lw=1.6, label="截斷保留 2 位（4 次）")],
                      loc="upper right", bbox_to_anchor=(0.79, 1.0))
    ax.add_artist(kinds)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.1, label="seed 1234"),
                       Line2D([], [], color=INK2, lw=1.1, ls=dashed, label="seed 5678")],
              loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ticks = sorted({q[0] for p in curves.values() for q in p})
    ax.set_xticks(ticks)
    ax.set_xlim(ticks[0] - 120, ticks[-1] + 120)
    ax.set_ylim(0.919, 0.9525)
    ax.set_yticks([0.92, 0.93, 0.94, 0.95])
    ax.set_xlabel("訓練樣本數")
    ax.set_ylabel("驗證損失")
    save(fig, "fig5_10_lora.pdf")


# ----------------------------------------------------------------------------------------------- 5.11
def fig_ppl():
    """5.11 WikiText-2 perplexity increase over native BF16 (sequence length 2048), weights and activations.
    infer_ppl.csv (jobs 1817, 1823, 1827, 1829-1831, 1837; Llama-2-70B evaluated layer by layer)."""
    d = pd.read_csv(res("infer_ppl.csv"))
    d = d[(d.wa == "wa") & d.attn.isna() & (d.seqlen == 2048) & ~d.spec.str.contains(";")]
    models = [("Llama-2-7b-hf", "Llama-2-7B"), ("Meta-Llama-3.1-8B", "Llama-3.1-8B"), ("Qwen2.5-7B", "Qwen2.5-7B"),
              ("Mistral-7B-v0.1", "Mistral-7B"), ("opt-6.7b", "OPT-6.7B"), ("Llama2-70B", "Llama-2-70B")]
    # blue = the mantissa-reduced formats (the subject), grey = the FP8 baselines
    fmts = [("bf16:3", "BF16 最近捨入保留 3 位", dict(marker="o", color=BLUE, ms=4.6, mec="white", mew=0.5), 0.22),
            ("fp8", "FP8 逐張量縮放", dict(marker="s", color=GREY_D, ms=4.0, mec="white", mew=0.5), 0.0),
            ("fp8c", "FP8 逐通道縮放", dict(marker="D", color=GREY_L, ms=3.8, mec="white", mew=0.5), -0.22),
            ("fp8c:2", "FP8 逐通道縮放再截斷至 2 位",
             dict(marker="^", color=BLUE, ms=5.0, mfc="white", mec=BLUE, mew=1.0), 0.0)]
    fig, ax = fig_axes(FULL, 7.2, 2.05, 0.3, 1.0, 1.15)
    for fmt, lab, kw, dy in fmts:
        xs, ys = [], []
        for i, (m, _) in enumerate(models):
            g = d[(d.model == m) & (d.fmt == fmt)]
            assert len(g) == 1, (m, fmt)
            xs.append(g.delta_pct.iloc[0])
            ys.append(len(models) - 1 - i + dy)
        ax.plot(xs, ys, ls="none", label=lab, zorder=3, **kw)
        if fmt == "fp8c:2":
            for x, y in zip(xs, ys):
                ax.text(x + 0.15, y, pct(x), ha="left", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(models)), [lab for _, lab in models][::-1])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_ylim(-0.6, len(models) - 0.4)
    ax.set_xlim(0, 8)
    ax.set_xticks(range(9))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WikiText-2 困惑度相對原生 BF16 的增加（%，越左越好）")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.13, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.3,
              columnspacing=2.0)
    save(fig, "fig5_11_ppl.pdf")


# ----------------------------------------------------------------------------------------------- 5.12
def fig_spread():
    """5.12 Synthetic exponent spread within rows (8192x8192x16384, 1000 W; errors are identical at 575 W):
    median elementwise relative error. range_sweep.csv (jobs 2034-2036). S >= 16 is on a log-2 spacing; S = 0 is
    placed one step to the left behind an axis break."""
    d = pd.read_csv(res("range_sweep.csv"))
    d = d[(d.smode == "elem") & (d.tdp == 1000)]
    ss = sorted(d.spread.unique())
    series = [(("fp64", 36), "截斷 36 位", dict(color=BLUE, marker="o", ms=4.2, lw=1.6), 1.0),
              (("fp64emu", 32), "cuBLAS 模擬 32 位", dict(color=GREY_D, marker="s", ms=3.8), 0.62),
              (("gemmul8", 10), "GEMMul8 10 模數", dict(color=GREY, marker="D", ms=3.4, ls=(0, (3.5, 2))), 2.1),
              (("fp64emu", 40), "cuBLAS 模擬 40 位", dict(color=GREY_D, marker="s", ms=3.8, mfc="white"), 1.0),
              (("fp32", 23), "FP32", dict(color=GREY_L, marker="v", ms=4.2), 1.0)]
    fig, ax = fig_axes(FULL, 7.2, 1.45, 3.0, 1.0, 0.2)
    xs = range(len(ss))
    for (dt, fr), lab, kw, nudge in series:
        g = d[(d.dtype == dt) & (d.frac == fr)].set_index("spread").reindex(ss)
        fin = [(i, v) for i, v in zip(xs, g.err_p50) if np.isfinite(v)]
        kw = dict(kw)
        kw.setdefault("lw", 1.3)
        ax.plot([p[0] for p in fin], [p[1] for p in fin], zorder=3 if dt == "fp64" else 2, **kw)
        last_i, last_v = fin[-1]
        if last_i == len(ss) - 1:
            ax.annotate(lab, (last_i, last_v), xytext=(last_i + 0.22, last_v * nudge), textcoords="data", ha="left",
                        va="center", fontsize=8, color=INK,
                        arrowprops=None if nudge == 1.0 else dict(arrowstyle="-", color=MUTED, lw=0.5,
                                                                   shrinkA=0, shrinkB=3))
        else:  # FP32 overflows at the largest spread
            bad = [s for s, v in zip(ss, g.err_p50) if not np.isfinite(v)]
            ax.text(last_i + 0.15, last_v, f"{lab}（S = {int(bad[0])} 時溢位）", ha="left", va="center",
                    fontsize=8, color=INK)
    # axis break between S = 0 and S = 16 (0 cannot sit on the log-2 spacing)
    tr = ax.get_xaxis_transform()
    ax.add_patch(Rectangle((0.44, -0.03), 0.12, 0.06, transform=tr, fc="white", ec="none", clip_on=False, zorder=5))
    for xb in (0.44, 0.56):
        ax.plot([xb - 0.035, xb + 0.035], [-0.03, 0.03], transform=tr, color=SPINE, lw=0.6, clip_on=False, zorder=6)
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 1e-5)
    ax.set_yticks([10.0 ** e for e in range(-15, -4, 2)])
    ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=20))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks(list(xs), [str(int(s)) for s in ss])
    ax.set_xlim(-0.25, len(ss) - 0.75)
    ax.set_xlabel("列內逐元素的指數跨度 S（以 2 的冪次計，對數間距）")
    ax.set_ylabel("逐元素相對誤差的中位數")
    save(fig, "fig5_12_spread.pdf")


# ----------------------------------------------------------------------------------------------- 5.13
def fig_rangemap():
    """5.13 Global exponent span vs fraction of elements smaller than 2^-32 times their row maximum, coloured by
    whether fixed-bit INT8 emulation was correct. range_map_points.csv (sources listed per point). Labels stay on
    their point's side of the format-range lines."""
    t = pd.read_csv(res("range_map_points.csv"))
    # (offset in points, horizontal alignment, vertical alignment); two-line labels split at '（' or before 'DFT'
    off = {"HPL (uniform)": ((14, 6), "right", "bottom"), "DF-MP2 tensor": ((0, 6), "center", "bottom"),
           "QE Au surface": ((-7, 0), "right", "center"), "QE Si": ((-7, 0), "right", "center"),
           "Gaussian-basis DFT": ((-7, -3), "right", "top"), "GP geostatistics": ((-7, 0), "right", "center")}
    fig, ax = fig_axes(FULL, 7.6, 2.3, 0.3, 1.0, 0.75)
    for x, lab in ((18, "FP8 E4M3"), (40, "FP16"), (254, "FP32/BF16")):
        ax.axvline(x, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.text(x * 1.06, 7e-7, f"{lab}\n的範圍", ha="left", va="bottom", fontsize=8, color=INK2, linespacing=1.15)
    for r in t.itertuples():
        c = BLUE if bool(r.emu_ok) else ORANGE
        ax.plot([r.global_span], [r.frac_below32], ls="none", marker="o", ms=6, color=c, mec="white", mew=0.7,
                zorder=3)
        lab = r.label_zh.replace("（", "\n（") if "（" in r.label_zh else r.label_zh.replace(" DFT", "\nDFT")
        xy, ha, va = off[r.label_en]
        ax.annotate(lab, (r.global_span, r.frac_below32), textcoords="offset points", xytext=xy, ha=ha, va=va,
                    fontsize=8, color=INK, zorder=4, linespacing=1.15, multialignment=ha if ha != "center" else "center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(8, 1024)
    ax.set_ylim(5e-7, 3)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024], ["8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1], ["0.0001%", "0.01%", "1%", "100%"])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("整體指數跨度（中間 99.9% 的數值，以 2 的冪次計）")
    ax.set_ylabel("小於該列最大值 $2^{-32}$ 倍的\n元素比例（依運算量加權）")
    wrong = t[~t.emu_ok.astype(bool)].emu_outcome.str.extract(r"([\d.]+) mEh")[0].iloc[0]  # '52'
    handles = [Line2D([], [], ls="none", marker="o", ms=6, color=BLUE, mec="white", label="固定位數的 INT8 模擬結果可用"),
               Line2D([], [], ls="none", marker="o", ms=6, color=ORANGE, mec="white",
                      label=rf"固定位數的 INT8 模擬靜默算錯（{wrong} m$E_{{\mathrm{{h}}}}$）")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2,
              handletextpad=0.3)
    save(fig, "fig5_13_rangemap.pdf")


FIGS = [fig_mechanism, fig_gemm_caps, fig_hpl, fig_hplmxp, fig_qe_rounding, fig_emu_caps, fig_shapes, fig_crossover,
        fig_nongemm, fig_lora, fig_ppl, fig_spread, fig_rangemap]


def main():
    global OUT
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "paper", "zh_report", "figs")
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        setup_style(setup_fonts(sys.argv[1], tmp))
        for f in FIGS:
            f()


if __name__ == "__main__":
    main()
