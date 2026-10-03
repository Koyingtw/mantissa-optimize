"""Gaussian-process (geostatistics) log-likelihood with a blocked Cholesky on the GPU, the kernel of
ExaGeoStat-style maximum-likelihood estimation: build an n x n Matern-3/2 covariance for n random 2D
locations, factor it, and evaluate the log-likelihood at several range parameters. The factorisation is
a right-looking blocked Cholesky whose trailing updates are one FP64 DGEMM per panel (like HPL), called
through cuBLAS with ctypes on cupy's handle, so the panel operand can be mantissa-reduced in-program and
the handle can be switched to cuBLAS FP64 emulation.

usage: python gp_mle.py            (env below)
  GP_N (60000), GP_NB (1024), GP_RANGES ("0.03 0.05 0.08"), GP_NUGGET (1e-4), GP_SEED (1)
  GP_VARIANTS: native | t<k><n|t> | emu<k> | emud | fp32   (space separated, run in order, same data)
  GP_CAP: module power cap to check before each variant (re-applied if another user's tool reset it)
Prints one RESULT_GP line per variant with the log-likelihoods, their difference from the first native
run, wall time and Unix start/end; RANGE_GP lines give the per-row exponent span of the panel operand.
"""
import ctypes
import math
import os
import socket
import subprocess
import time

import cupy as cp
import numpy as np

N = int(os.environ.get("GP_N", 60000))
NB = int(os.environ.get("GP_NB", 1024))
RANGES = [float(x) for x in os.environ.get("GP_RANGES", "0.03 0.05 0.08").split()]
NUGGET = float(os.environ.get("GP_NUGGET", 1e-4))

_lib = None


def lib():
    global _lib
    if _lib is None:
        cp.cuda.device.get_cublas_handle()
        path = next(l.split()[-1] for l in open('/proc/self/maps') if '/libcublas.so' in l)
        _lib = ctypes.CDLL(path)
    return _lib


def handle():
    return ctypes.c_void_p(cp.cuda.device.get_cublas_handle())


def set_emulation(bits):
    h, L = handle(), lib()
    if bits is None:
        assert L.cublasSetMathMode(h, 0) == 0 and L.cublasSetEmulationStrategy(h, 0) == 0
        return
    assert L.cublasSetMathMode(h, 8) == 0 and L.cublasSetEmulationStrategy(h, 2) == 0
    assert L.cublasSetFixedPointEmulationMantissaControl(h, 0 if bits == 0 else 1) == 0
    if bits > 0:
        assert L.cublasSetFixedPointEmulationMaxMantissaBitCount(h, bits) == 0


def reduce_(x, bits, nearest):
    if bits >= 52:
        return
    u = x.view(cp.uint64)
    drop = 52 - bits
    if nearest:
        u += cp.uint64(1 << (drop - 1))
    u &= cp.uint64(((1 << 64) - 1) ^ ((1 << drop) - 1))


def covariance(xy, rng, dtype):
    """Matern-3/2 covariance, column-major (order F), built in column blocks to bound temporaries."""
    n = xy.shape[0]
    K = cp.empty((n, n), dtype=dtype, order='F')
    s = math.sqrt(3.0) / rng
    for j in range(0, n, 4096):
        d = cp.sqrt(((xy[:, None, :] - xy[None, j:j + 4096, :]) ** 2).sum(-1))
        K[:, j:j + 4096] = ((1 + s * d) * cp.exp(-s * d)).astype(dtype)
        del d
    K[cp.arange(n), cp.arange(n)] += NUGGET
    return K


SPAN = {}


def span_stats(Lt, nrows=256):
    r = cp.abs(Lt[:: max(1, Lt.shape[0] // nrows)])
    e = cp.where(r > 0, cp.log2(cp.where(r > 0, r, 1)), cp.nan)
    mx = cp.nanmax(e, axis=1)[:, None]
    below = mx - e
    ok = ~cp.isnan(below)
    n = int(ok.sum())
    for t in (24, 32, 40):
        SPAN.setdefault(t, [0, 0])
        SPAN[t][0] += int(((below > t) & ok).sum())
        SPAN[t][1] += n


def cholesky(A, bits, nearest, fp32, stats=False):
    """In-place lower blocked Cholesky of the F-ordered A; the panel L21 is copied (and reduced) before the
    trailing DGEMM A22 -= L21 L21^T, which runs on the full square block (only the lower part is used)."""
    n = A.shape[0]
    h, L = handle(), lib()
    esz = 4 if fp32 else 8
    gemm = L.cublasSgemm_v2 if fp32 else L.cublasDgemm_v2
    trsm = L.cublasStrsm_v2 if fp32 else L.cublasDtrsm_v2
    scal = ctypes.c_float if fp32 else ctypes.c_double
    one, mone = scal(1.0), scal(-1.0)
    base = A.data.ptr
    ptr = lambda i, j: ctypes.c_void_p(base + (i + j * n) * esz)  # noqa: E731
    for j in range(0, n, NB):
        b = min(NB, n - j)
        D = A[j:j + b, j:j + b]
        Ld = cp.linalg.cholesky(cp.ascontiguousarray(D))
        if bool(cp.isnan(Ld).any()):
            raise np.linalg.LinAlgError(f"not positive definite at column {j}")
        A[j:j + b, j:j + b] = Ld
        r = n - j - b
        if r == 0:
            break
        # L21 = A21 L11^-T  (side right, lower, transpose, non-unit)
        assert trsm(h, 1, 0, 1, 0, r, b, ctypes.byref(one), ptr(j, j), n, ptr(j + b, j), n) == 0
        Lt = cp.asfortranarray(A[j + b:, j:j + b])
        if not fp32:
            reduce_(Lt, bits, nearest)
            if stats and j % (8 * NB) == 0:
                span_stats(Lt)
        assert gemm(h, 0, 1, r, r, b, ctypes.byref(mone), ctypes.c_void_p(Lt.data.ptr), r,
                    ctypes.c_void_p(Lt.data.ptr), r, ctypes.byref(one), ptr(j + b, j + b), n) == 0
    cp.cuda.Device().synchronize()


def loglik(A, y):
    d = cp.diagonal(A).astype(cp.float64)
    logdet = 2.0 * float(cp.log(d).sum())
    import cupyx.scipy.linalg as sl
    z = sl.solve_triangular(A, y.astype(A.dtype), lower=True)
    q = float((z.astype(cp.float64) ** 2).sum())
    return -0.5 * (q + logdet + A.shape[0] * math.log(2 * math.pi))


def smi(field):
    return subprocess.run(['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.strip()


def wait_cool(limit=45.0, max_wait=180):
    t = time.time()
    while float(smi('temperature.gpu')) > limit and time.time() - t < max_wait:
        time.sleep(5)


def ensure_cap():
    want = os.environ.get('GP_CAP')
    if want and smi('module.power.limit').split('.')[0] != want:
        print(f"CAP RESET detected -> re-applying {want} W", flush=True)
        subprocess.run(['nv_power', '--hosts', socket.gethostname(), want], capture_output=True)


rs = np.random.default_rng(int(os.environ.get("GP_SEED", 1)))
xy = cp.asarray(rs.random((N, 2)))
# observations drawn from the model at the middle range (factor once in native FP64)
K0 = covariance(xy, RANGES[len(RANGES) // 2], cp.float64)
cholesky(K0, 52, True, False)
y = cp.tril(K0[:, :]) @ cp.asarray(rs.standard_normal(N)) if N <= 20000 else None
if y is None:  # matrix-vector with the factor without forming tril(K0): blocked
    zv = cp.asarray(rs.standard_normal(N))
    y = cp.zeros(N)
    for j in range(0, N, 8192):
        y += cp.tril(K0[:, j:j + 8192], k=-j) @ zv[j:j + 8192]
del K0
cp.get_default_memory_pool().free_all_blocks()
print(f"n={N} nb={NB} ranges={RANGES} nugget={NUGGET}", flush=True)

ref = None
seen = {}
for v in os.environ.get("GP_VARIANTS", "native").split():
    seen[v] = seen.get(v, 0) + 1
    pas = "ABCDEFGH"[seen[v] - 1]
    bits, nearest, fp32, emu = 52, True, False, None
    if v.startswith("t"):
        bits, nearest = int(v[1:-1]), v[-1] == "n"
    elif v == "emud":
        emu = 0
    elif v.startswith("emu"):
        emu = int(v[3:])
    elif v == "fp32":
        fp32 = True
    wait_cool()
    ensure_cap()
    set_emulation(emu)
    t0 = time.time()
    lls = []
    status = "ok"
    for rng in RANGES:
        A = covariance(xy, rng, cp.float32 if fp32 else cp.float64)
        try:
            cholesky(A, bits, nearest, fp32, stats=(v.startswith("t") or v == "native") and seen[v] == 1)
            lls.append(loglik(A, y))
        except Exception as e:  # noqa: BLE001
            status = f"failed({type(e).__name__})"
            lls.append(float("nan"))
        del A
        cp.get_default_memory_pool().free_all_blocks()
    t1 = time.time()
    set_emulation(None)
    if v == "native" and ref is None:
        ref = lls
    d = [l - r for l, r in zip(lls, ref)] if ref else [float("nan")] * len(lls)
    print(f"RESULT_GP tdp={os.environ.get('GP_CAP', '-')} variant={v} pass={pas} status={status} "
          f"loglik=[{' '.join(f'{x:.6f}' for x in lls)}] dll=[{' '.join(f'{x:.3e}' for x in d)}] "
          f"wall={t1 - t0:.3f}s t_begin={t0:.3f} t_end={t1:.3f}", flush=True)
    if SPAN:
        print("RANGE_GP " + " ".join(f"frac_below{t}={a / max(b, 1):.3e}" for t, (a, b) in sorted(SPAN.items())), flush=True)
        SPAN.clear()
