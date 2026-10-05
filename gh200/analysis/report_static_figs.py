"""Static PNG versions of the 專題報告's figures 5.9-5.13 (the report's PDF export renders only its first eight
interactive charts). Data are the same values the interactive charts carry.

usage: report_static_figs.py <font.ttf with Traditional Chinese glyphs> [outdir]   (outdir default: report_figs)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FONT, OUT = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "report_figs")
os.makedirs(OUT, exist_ok=True)
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(), "font.size": 10, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e6eb",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.unicode_minus": False, "figure.dpi": 100,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a6f97", "#d9792b", "#4f9a6b", "#8d6cab", "#6b7585"
W, H = 7.6, 4.1


def titled(fig, title, sub, top=0.83):
    fig.subplots_adjust(top=top)
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=12.5)
    fig.text(0.02, 0.895, sub, ha="left", fontsize=9, color=GRAY)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def fig_nongemm():
    rows = [("乘加迴圈", 862, 62, 138), ("記憶體複製", 672, 31, 70), ("triad（讀兩寫一）", 706, 31, 75),
            ("N 體（只截輸入）", 775, 28, 59), ("CG 稀疏求解", 751, 0, 7)]
    fig, ax = plt.subplots(figsize=(W, H))
    x = range(len(rows))
    for j, (bits, alpha) in enumerate(((36, 0.45), (20, 1.0))):
        vals = [r[2] if bits == 36 else r[3] for r in rows]
        bars = ax.bar([i - 0.19 + 0.38 * j for i in x], vals, width=0.36, color=BLUE, alpha=alpha,
                      label=f"保留 {bits} 位")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, "0" if v == 0 else f"−{v}", ha="center", va="bottom",
                    fontsize=9, color=BLUE if bits == 20 else GRAY, fontweight="bold" if bits == 20 else "normal")
    ax.set_xticks(list(x), [f"{r[0]}\n原生 {r[1]} W" for r in rows], fontsize=9)
    ax.set_ylim(0, 160)
    ax.set_yticks([0, 50, 100, 150], ["0", "−50 W", "−100 W", "−150 W"])
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    titled(fig, "不只 tensor core：乘加迴圈與記憶體搬移也省電，CG 幾乎沒有",
           "1000 W（時脈都在 1980 MHz，效能不變），模組功耗比原生少多少瓦")
    save(fig, "fig5_9_nongemm.png")


def fig_lora():
    runs = {
        "n575": ("native", "-", [(1536, .9396), (1920, .9361), (2304, .9313), (2688, .9295), (3072, .9255), (3456, .9231)]),
        "n800a": ("native", "-", [(1536, .9404), (1920, .9361), (2304, .935), (2688, .9277), (3072, .9254), (3456, .9234)]),
        "n800b": ("native", "-", [(1536, .94), (1920, .9365), (2304, .9308), (2688, .9276), (3072, .9237)]),
        "n5678": ("native", "--", [(1536, .9393), (1920, .9338), (2304, .9329), (2688, .9271), (3072, .9225)]),
        "t575": ("trunc", "-", [(1536, .9479), (1920, .9433), (2304, .938), (2688, .9367), (3072, .9316), (3456, .9293),
                                (3840, .9272), (4224, .9246)]),
        "t800a": ("trunc", "-", [(1536, .9483), (1920, .9431), (2304, .9379), (2688, .9344), (3072, .9336), (3456, .9323),
                                 (3840, .9267), (4224, .9245)]),
        "t800b": ("trunc", "-", [(1536, .9486), (1920, .944), (2304, .9386), (2688, .9356), (3072, .9338), (3456, .931),
                                 (3840, .9271), (4224, .9248)]),
        "t5678": ("trunc", "--", [(1536, .9491), (1920, .9473), (2304, .9409), (2688, .9365), (3072, .9333), (3456, .9294),
                                  (3840, .9284), (4224, .9248)]),
    }
    fig, ax = plt.subplots(figsize=(W, H))
    for name, (kind, ls, pts) in runs.items():
        c = BLUE if kind == "native" else ORANGE
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker="o", ms=3.5, lw=1.8, color=c, alpha=0.85)
    ax.axhline(0.925, color=GRAY, ls=(0, (5, 4)), lw=1)
    ax.text(1560, 0.9254, "達標 0.925", ha="left", va="bottom", fontsize=9, color=GRAY)
    ax.plot([], [], color=BLUE, lw=2.5, label="原生 FP8（4 次）")
    ax.plot([], [], color=ORANGE, lw=2.5, label="截斷保留 2 位（4 次）")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xticks([1536, 1920, 2304, 2688, 3072, 3456, 3840, 4224])
    ax.set_xlabel("訓練樣本數", color=GRAY)
    ax.set_ylim(0.92, 0.952)
    titled(fig, "兩個 seed 都一樣：截斷後驗證損失從第一次評估就上移，多 2–3 個評估區間",
           "Llama 2 70B LoRA，8 節點；實線 seed 1234、虛線 seed 5678；水平虛線為達標門檻 0.925")
    save(fig, "fig5_10_lora.png")


def fig_ppl():
    data = {"Llama-2-7B": (0.44, 0.63, 0.32, 3.2), "Llama-3.1-8B": (1.02, 0.95, 0.88, 6.5),
            "Qwen2.5-7B": (0.65, 0.70, 0.61, 4.0), "Mistral-7B": (0.41, 0.45, 0.34, 2.5),
            "OPT-6.7B": (2.00, 2.23, 0.69, 7.0), "Llama2-70B": (0.51, 0.48, 0.42, 3.4)}
    fmts = [("BF16 最近捨入保留 3 位", BLUE, -0.12), ("FP8 逐張量", ORANGE, 0.0), ("FP8 逐通道", GREEN, 0.12),
            ("FP8 逐通道再截到 2 位", PURPLE, 0.0)]
    fig, ax = plt.subplots(figsize=(W, H))
    models = list(data)
    for j, (lab, c, dy) in enumerate(fmts):
        ys = [len(models) - 1 - i + dy for i in range(len(models))]
        xs = [data[m][j] for m in models]
        ax.scatter(xs, ys, s=38, color=c, label=lab, zorder=3)
        if j == 3:
            for x, y in zip(xs, ys):
                ax.text(x + 0.15, y, f"+{x}%", va="center", fontsize=9, color=GRAY)
    ax.set_yticks(range(len(models)), models[::-1])
    ax.set_xlim(0, 7.9)
    ax.set_xticks(range(8), [f"+{t}%" for t in range(8)])
    ax.set_xlabel("困惑度增加（越左越好）", color=GRAY)
    ax.grid(axis="y", ls=(0, (2, 3)))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=9, handletextpad=0.2,
              columnspacing=1.0)
    titled(fig, "BF16 保留 3 位與 FP8 精度相當；FP8 再截到 2 位就明顯變差",
           "WikiText-2 困惑度（序列長度 2048）相對 BF16 原生的增加百分比；FP8 用真實的 FP8 tensor core 矩陣乘法")
    save(fig, "fig5_11_ppl.png")


def fig_spread():
    ss = [0, 16, 32, 64, 128, 256]
    series = [("截斷 FP64 36 位", BLUE, [1.02e-11, 1.13e-11, 1.13e-11, 1.12e-11, 1.11e-11, 1.08e-11]),
              ("cuBLAS 模擬 32 位", ORANGE, [9.71e-12, 3.94e-11, 5.19e-11, 5.35e-11, 5.96e-11, 8.95e-11]),
              ("cuBLAS 模擬 40 位", GREEN, [4.06e-14, 1.74e-13, 2.52e-13, 2.69e-13, 3.00e-13, 4.51e-13]),
              ("GEMMul8 10 模數", PURPLE, [4.12e-10, 2.04e-10, 2.84e-10, 1.55e-10, 1.73e-10, 1.36e-10]),
              ("FP32", "#b5485a", [1.2e-6, 1.11e-6, 6.54e-7, 3.32e-7, 1.66e-7, None])]
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(right=0.78)
    xs = range(len(ss))
    for lab, c, vals in series:
        pts = [(i, v) for i, v in zip(xs, vals) if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, lw=2.2, color=c)
        nudge = {"GEMMul8 10 模數": 1.9, "cuBLAS 模擬 32 位": 0.55}.get(lab, 1.0)
        ax.text(len(ss) - 1 + 0.15, pts[-1][1] * nudge, lab, va="center", fontsize=9, color=c)
        if None in vals:
            ax.text(len(ss) - 1, 4e-6, "溢位", ha="center", fontsize=10, color=c, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1e-15, 1e-5)
    ax.set_xticks(list(xs), [str(s) for s in ss])
    ax.set_xlabel("逐元素指數跨度 S（2 的次方數）", color=GRAY)
    titled(fig, "跨度變大時，截斷的誤差不變，模擬變差、FP32 溢位",
           "8192×8192×16384，每個元素的指數在 S 個 2 的次方內均勻分佈；縱軸為逐元素相對誤差的中位數")
    save(fig, "fig5_12_spread.png")


def fig_map():
    pts = [("HPL（均勻亂數）", 14, 1e-6, True, (8, -10)), ("DF-MP2 張量（紫杉醇）", 27, 2.4e-5, True, (8, 12)),
           ("QE 金表面", 93, 2.2e-4, True, (-8, -10)), ("QE 矽", 91, 5.4e-3, True, (-8, -10)),
           ("PySCF 纈胺黴素 DFT", 690, 0.44, False, (-8, -12)), ("高斯過程地統計（n = 80000）", 150, 0.88, True, (-8, 14))]
    fig, ax = plt.subplots(figsize=(W, 4.6))
    for x, lab in [(18, "FP8 E4M3"), (40, "FP16"), (254, "FP32／BF16")]:
        ax.axvline(x, color=GRAY, lw=0.9, ls=(0, (5, 4)))
        ax.text(x * 1.05, 0.7e-6, f"{lab} 的範圍", fontsize=8.5, color=GRAY, va="bottom")
    for lab, g, r, ok, (dx, dy) in pts:
        ax.scatter([g], [r], s=60, color=BLUE if ok else ORANGE, zorder=3)
        ax.annotate(lab, (g, r), textcoords="offset points", xytext=(dx, -dy), fontsize=9,
                    ha="left" if dx > 0 else "right", va="center")
    ax.scatter([], [], s=50, color=BLUE, label="模擬結果可用（GP 誤差大 100 倍、較慢）")
    ax.scatter([], [], s=50, color=ORANGE, label="模擬安靜地算錯（52 mEh）")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(8, 1024)
    ax.set_ylim(5e-7, 3)
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024], ["8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1], ["≤0.0001%", "0.01%", "1%", "100%"])
    ax.set_xlabel("整體指數跨度（中間 99.9% 的數值橫跨幾個 2 的次方，對數尺度）", color=GRAY)
    ax.set_ylabel("列內比最大值小 $2^{-32}$ 以上的元素\n（依運算量加權）", color=GRAY)
    titled(fig, "整體跨度決定逐張量格式放不放得下，列內跨度決定 INT8 模擬會不會算錯",
           "截斷保留每個元素自己的指數，在圖上任何位置都可用；顏色是 cuBLAS 模擬 32–36 位在該應用上的實測結果")
    save(fig, "fig5_13_rangemap.png")


if __name__ == "__main__":
    for f in (fig_nongemm, fig_lora, fig_ppl, fig_spread, fig_map):
        f()
