"""MI350X 功耗／時脈取樣器：每 INTERVAL 秒讀一次 8 張 GPU 的 gpu_metrics，寫成一個 CSV。

usage: smi_sampler.py <out.csv> [interval_s=0.1]   （SIGTERM／SIGINT 結束）

欄位：t（Unix 秒）、gpu（HIP 裝置序＝amd-smi 序，依 BDF 排序）、power_w（current_socket_power，PPT
控制器所看的 socket 功耗）、gfxclk（8 個 XCC 的平均 MHz）、gfxclk_min、uclk、t_hot、t_mem、
energy_acc（硬體能量累計器原始值，單位見 ENERGY_UNIT_J）、ppt_acc／thm_acc（PPT 與溫度節流的
駐留累計，thm = socket + VR + HBM）、acc（這些駐留計數的分母 accumulation_counter）、busy（gfx 活動 %）、
cap_w（當下生效的 PPT0 上限，每秒讀一次）。
"""
import signal
import sys
import time

import amdsmi as A

out = sys.argv[1]
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

A.amdsmi_init()
hs = A.amdsmi_get_processor_handles()
hs = [h for _, h in sorted((A.amdsmi_get_gpu_device_bdf(h), h) for h in hs)]


def num(v):
    return v if isinstance(v, (int, float)) else -1


caps = [0] * len(hs)
last_cap = 0.0
with open(out, "w", buffering=1 << 16) as f:
    f.write("t,gpu,power_w,gfxclk,gfxclk_min,uclk,t_hot,t_mem,energy_acc,ppt_acc,thm_acc,acc,busy,cap_w\n")
    while not stop:
        t0 = time.time()
        if t0 - last_cap >= 1.0:
            caps = [A.amdsmi_get_power_cap_info(h)["power_cap"] // 1000000 for h in hs]
            last_cap = t0
        for i, h in enumerate(hs):
            m = A.amdsmi_get_gpu_metrics_info(h)
            clks = [c for c in m["current_gfxclks"] if isinstance(c, int)]
            thm = sum(num(m[k]) for k in ("socket_thm_residency_acc", "vr_thm_residency_acc", "hbm_thm_residency_acc"))
            f.write(f"{time.time():.3f},{i},{num(m['current_socket_power'])},{sum(clks) / len(clks):.0f},{min(clks)},"
                    f"{num(m['current_uclk'])},{num(m['temperature_hotspot'])},{num(m['temperature_mem'])},"
                    f"{m['energy_accumulator']},{num(m['ppt_residency_acc'])},{thm},{num(m['accumulation_counter'])},"
                    f"{num(m['average_gfx_activity'])},{caps[i]}\n")
        f.flush()
        time.sleep(max(0.0, interval - (time.time() - t0)))
A.amdsmi_shut_down()
