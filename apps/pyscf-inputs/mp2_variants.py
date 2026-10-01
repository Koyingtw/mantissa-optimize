# 用法: python mp2_variants.py <xyz> [basis]
# 在同一個程序裡先做一次 DF-RHF，再依 MP2_VARIANTS 依序做 DF-MP2，所以各變體的 HF 能量與軌域完全相同。
# DF-MP2 的時間幾乎都花在 cderi_ovl[i] @ cderi_ovl[j].T（FP64 DGEMM，k = 輔助基底數）。
# 變體：
#   native     原生 FP64
#   fp32       GPU4PySCF 內建的 FP32 張量與 SGEMM（換型別的做法）
#   t<k><n|t>  DF 張量（cderi_ovl）建好後保留 k 個尾數位元，n = 最近捨入、t = 截斷；GEMM 仍是 FP64
#   emu<k>     cuBLAS FP64 定點模擬（INT8 tensor core），固定保留 k 位
#   emud       cuBLAS FP64 定點模擬，動態決定位數（保證不差於 FP64）
# 每個變體印一行 RESULT_MP2（含 Unix 起訖時間，用來對應 nvidia-smi 功耗紀錄）；開始前等 GPU 降到 MP2_START_TEMP 以下，
# 並確認模組功耗上限仍是 MP2_CAP（別人的工具可能重設全叢集的上限）。
# MP2_RANGE_STATS=1：第一次建好 cderi_ovl 時印出指數跨度統計（整體與逐列，逐列 = GEMM 的一列，沿輔助基底）。
import ctypes
import os
import socket
import subprocess
import sys
import time

import cupy
import numpy as np
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.mp import dfmp2
import gpu4pyscf.mp.dfmp2_addons as _A

# cupy>=14 沒有 cuTENSOR → contract 走 cupy.einsum（→ cuBLAS），會多出暫存陣列；
# 原本 j3c batch 以 80% 可用記憶體估算會 OOM，這裡降為 MP2_MEM_PREFACTOR（預設 0.3）
_orig_est = _A.estimate_j3c_batch
_A.estimate_j3c_batch = lambda *a, **k: _orig_est(*a, **{**k, 'prefactor': float(os.environ.get('MP2_MEM_PREFACTOR', 0.3))})

STATE = {'bits': 52, 'nearest': True, 'stats_done': False}


def reduce_inplace(x, bits, nearest):
    """Keep `bits` of the 52 fraction bits of a float64 cupy array, in place."""
    if bits >= 52:
        return
    u = x.view(cupy.uint64)
    drop = 52 - bits
    if nearest:
        u += cupy.uint64(1 << (drop - 1))
    u &= cupy.uint64(((1 << 64) - 1) ^ ((1 << drop) - 1))


def range_stats(x, nrows=4096):
    """Exponent span of the GEMM operand: whole tensor and per row (a row of cderi_ovl[i] runs over the
    auxiliary index, which is what per-row scaled INT8 emulation normalises)."""
    rows = x.reshape(-1, x.shape[-1])
    pick = np.sort(np.random.default_rng(0).choice(rows.shape[0], min(nrows, rows.shape[0]), replace=False))
    r = np.abs(cupy.asnumpy(rows[cupy.asarray(pick)]))
    with np.errstate(divide='ignore'):
        e = np.log2(np.where(r > 0, r, np.nan))
    flat = e[~np.isnan(e)]
    lo, hi = np.percentile(flat, [0.05, 99.95])
    row_lo = np.nanpercentile(e, 0.05, axis=1)
    row_hi = np.nanpercentile(e, 99.95, axis=1)
    span = row_hi - row_lo
    below_row_max = (np.nanmax(e, axis=1)[:, None] - e)   # binades below the row maximum
    below_row_max = below_row_max[~np.isnan(below_row_max)]
    print(f"RANGE global_span={hi - lo:.1f} global_log2=[{lo:.1f},{hi:.1f}] "
          f"row_span_p50={np.percentile(span, 50):.1f} row_span_p90={np.percentile(span, 90):.1f} "
          f"row_span_max={span.max():.1f} frac_32below_rowmax={np.mean(below_row_max > 32):.3e} "
          f"frac_40below_rowmax={np.mean(below_row_max > 40):.3e} zero_frac={np.mean(r == 0):.3e} "
          f"shape={tuple(x.shape)}", flush=True)


_orig_decomp = _A.decompose_j3c_gpu


def decompose_and_reduce(*args, **kwargs):
    out = _orig_decomp(*args, **kwargs)
    for arr in out:
        if os.environ.get('MP2_RANGE_STATS') and not STATE['stats_done'] and arr.dtype == cupy.float64:
            range_stats(arr)
            STATE['stats_done'] = True
        if arr.dtype == cupy.float64:
            reduce_inplace(arr, STATE['bits'], STATE['nearest'])
    return out


_A.decompose_j3c_gpu = decompose_and_reduce

# cuBLAS FP64 emulation on cupy's own handle (the handle every cupy matmul in this thread uses)
_cublas = None


def cublas():
    global _cublas
    if _cublas is None:
        path = next(l.split()[-1] for l in open('/proc/self/maps') if '/libcublas.so' in l)
        _cublas = ctypes.CDLL(path)
    return _cublas


def set_emulation(bits):
    """bits = None: native FP64; bits = 0: dynamic mantissa control; bits > 0: fixed."""
    h = ctypes.c_void_p(cupy.cuda.device.get_cublas_handle())
    lib = cublas()
    if bits is None:
        assert lib.cublasSetMathMode(h, 0) == 0            # CUBLAS_DEFAULT_MATH
        assert lib.cublasSetEmulationStrategy(h, 0) == 0   # CUBLAS_EMULATION_STRATEGY_DEFAULT
        return
    assert lib.cublasSetMathMode(h, 8) == 0                # CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
    assert lib.cublasSetEmulationStrategy(h, 2) == 0       # CUBLAS_EMULATION_STRATEGY_EAGER
    assert lib.cublasSetFixedPointEmulationMantissaControl(h, 0 if bits == 0 else 1) == 0
    if bits > 0:
        assert lib.cublasSetFixedPointEmulationMaxMantissaBitCount(h, bits) == 0


def smi(field):
    return subprocess.run(['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.strip()


def wait_cool(limit, max_wait=180):
    t = time.time()
    while float(smi('temperature.gpu')) > limit and time.time() - t < max_wait:
        time.sleep(5)


def ensure_cap():
    want = os.environ.get('MP2_CAP')
    if not want:
        return
    cur = smi('module.power.limit').split('.')[0]
    if cur != want:
        print(f"CAP RESET detected: limit {cur} W, expected {want} W -> re-applying", flush=True)
        subprocess.run(['nv_power', '--hosts', socket.gethostname(), want], capture_output=True)


basis = sys.argv[2] if len(sys.argv) > 2 else 'def2-tzvp'
mol = gto.M(atom=sys.argv[1], basis=basis, verbose=3, max_memory=200000)
print(f"natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} basis={basis}", flush=True)
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10
t0 = time.time()
ehf = mf.kernel()
cupy.cuda.Device().synchronize()
print(f"RESULT_HF converged={mf.converged} E={ehf:.10f} Eh wall={time.time() - t0:.2f}s t_begin={t0:.3f} t_end={time.time():.3f}",
      flush=True)

ref = None
seen = {}
for v in os.environ.get('MP2_VARIANTS', 'native').split():
    seen[v] = seen.get(v, 0) + 1
    pas = 'ABCDEFGH'[seen[v] - 1]
    STATE.update(bits=52, nearest=True)
    fp_type = 'FP64'
    emu = None
    if v == 'fp32':
        fp_type = 'FP32'   # the instance attribute: DFMP2 copies CONFIG_FP_TYPE at class definition
    elif v.startswith('t'):
        STATE.update(bits=int(v[1:-1]), nearest=v[-1] == 'n')
    elif v == 'emud':
        emu = 0
    elif v.startswith('emu'):
        emu = int(v[3:])
    elif v != 'native':
        sys.exit(f"unknown variant {v}")
    wait_cool(float(os.environ.get('MP2_START_TEMP', 45)))
    ensure_cap()
    cupy.get_default_memory_pool().free_all_blocks()
    set_emulation(emu)
    t1 = time.time()
    pt = dfmp2.DFMP2(mf)
    pt.fp_type = fp_type
    ecorr, _ = pt.kernel()
    cupy.cuda.Device().synchronize()
    t2 = time.time()
    set_emulation(None)
    ecorr = float(ecorr)
    if v == 'native' and ref is None:
        ref = ecorr
    de = f"{ecorr - ref:.3e}" if ref is not None else "nan"
    print(f"RESULT_MP2 tdp={os.environ.get('MP2_CAP', '-')} variant={v} pass={pas} Ecorr={ecorr:.12f} dE={de} wall={t2 - t1:.3f}s t_begin={t1:.3f} t_end={t2:.3f}",
          flush=True)
