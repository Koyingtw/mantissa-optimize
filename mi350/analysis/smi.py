"""把 MI350X 的 amdsmi 取樣 log（tools/smi_sampler.py）與量測窗對齊。

能量用硬體能量累計器（gpu_metrics energy_accumulator，單位 ENERGY_UNIT_J）在窗口兩端的差，
不是功耗取樣的積分；PPT／溫度節流駐留比例 = Δresidency / Δaccumulation_counter。
"""
import os

import numpy as np
import pandas as pd

ENERGY_UNIT_J = 15.259e-6  # 15.259 µJ／count（AMD gpu_metrics 規格；與平均功耗交叉驗證見 check_energy_unit）


def load_smi(path):
    """取樣 CSV；提交到 git 的版本以 xz 壓縮（<id>_smi.csv.xz），兩者都可。"""
    if not os.path.exists(path) and os.path.exists(path + ".xz"):
        path += ".xz"
    return pd.read_csv(path)


def window_stats(smi, gpu, t0, t1):
    """單一 GPU 在 [t0, t1] 的統計；t0/t1 為 Unix 秒。"""
    w = smi[(smi.gpu == gpu) & (smi.t >= t0) & (smi.t <= t1)]
    if len(w) < 3:
        return {}
    a, b = w.iloc[0], w.iloc[-1]
    dt = b.t - a.t
    dacc = b.acc - a.acc
    # 取樣器每秒才讀一次上限（cap_w 是快取值），換上限後第一個窗口的開頭最多 1 s 會看到舊值；
    # 上限本身在 RUN 之前已由 pcap.sh 讀回確認，所以上限檢查略過窗口開頭 1 s
    wc = w[w.t >= t0 + 1.0]
    wc = wc if len(wc) else w
    return {
        "power_w": w.power_w.mean(),
        "energy_j": (b.energy_acc - a.energy_acc) * ENERGY_UNIT_J * (t1 - t0) / dt,
        "power_acc_w": (b.energy_acc - a.energy_acc) * ENERGY_UNIT_J / dt,
        "gfxclk": w.gfxclk.mean(),
        "gfxclk_min": w.gfxclk_min.mean(),
        "t_hot_max": w.t_hot.max(),
        "t_mem_max": w.t_mem.max(),
        "ppt_frac": (b.ppt_acc - a.ppt_acc) / dacc if dacc > 0 else np.nan,
        "thm_frac": (b.thm_acc - a.thm_acc) / dacc if dacc > 0 else np.nan,
        "cap_min": wc.cap_w.min(),
        "cap_max": wc.cap_w.max(),
    }


def node_window_stats(smi, t0, t1, gpus=range(8)):
    """8 張 GPU 合計（能量、功耗相加；時脈、溫度取平均／最大）。"""
    rows = [window_stats(smi, g, t0, t1) for g in gpus]
    rows = [r for r in rows if r]
    d = pd.DataFrame(rows)
    return {
        "power_w": d.power_w.sum(), "power_acc_w": d.power_acc_w.sum(), "energy_j": d.energy_j.sum(),
        "gfxclk": d.gfxclk.mean(), "t_hot_max": d.t_hot_max.max(), "ppt_frac": d.ppt_frac.mean(),
        "thm_frac": d.thm_frac.mean(), "cap_min": d.cap_min.min(), "cap_max": d.cap_max.max(),
    }


def check_energy_unit(smi, gpu=0):
    """累計器換算的平均功耗 ÷ 取樣平均功耗；應接近 1。"""
    w = smi[smi.gpu == gpu]
    a, b = w.iloc[0], w.iloc[-1]
    return (b.energy_acc - a.energy_acc) * ENERGY_UNIT_J / (b.t - a.t) / w.power_w.mean()
