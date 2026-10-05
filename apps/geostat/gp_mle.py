"""Gaussian-process (geostatistics) log-likelihood with a blocked Cholesky on the GPU, the kernel of
ExaGeoStat-style maximum-likelihood estimation: build an n x n Matern-3/2 covariance for n random 2D
locations, factor it, and evaluate the log-likelihood at several range parameters. The factorisation is
a right-looking blocked Cholesky whose trailing updates are one FP64 DGEMM per panel (like HPL), called
through cuBLAS with ctypes on cupy's handle, so the panel operand can be mantissa-reduced in-program and
the handle can be switched to cuBLAS FP64 emulation.

usage: python gp_mle.py            (env below)
  GP_N (60000), GP_NB (1024), GP_RANGES ("0.03 0.05 0.08"), GP_NUGGET (1e-4), GP_SEED (1)
  GP_VARIANTS: native | t<k><n|t> | emu<k> | emud | fp32 | mp<D>[+t<k><n|t>][+h<k>]
               (space separated, run in order, same data). mp<D>|mpn is a mixed-precision factorisation in the
               style of ExaGeoStat's tile methods: trailing-update tiles within D tiles of the diagonal (mpn: chosen by
               the tile-norm rule, GP_MP_TOL) stay FP64, the others use FP16 inputs with FP32 accumulation on the
               tensor cores; +t<k><n|t> reduces the FP64 operand, +h<k> keeps k of the 10 FP16 mantissa bits
  GP_ORDER=morton: sort the locations along a Morton (Z-order) curve first, as tile methods require
  GP_MODE=mle: instead of the fixed ranges, estimate the range by golden-section maximisation of the
               log-likelihood on [GP_LO, GP_HI] (0.01, 0.2) to GP_TOL (0.002 in log10) per variant
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


# one fused pass per operand: add half an ulp of the kept bits (round to nearest, ties away) or nothing (truncate),
# then mask. The panels are column-major, so every elementwise pass runs on the transposed (row-major) view, which
# cupy walks in memory order; walking the column-major array in its index order is uncoalesced and several times slower
_RED64 = cp.ElementwiseKernel('uint64 add, uint64 mask', 'uint64 u', 'u = (u + add) & mask', 'gp_reduce64')
_RED16 = cp.ElementwiseKernel('uint16 add, uint16 mask', 'uint16 u', 'u = (u + add) & mask', 'gp_reduce16')


def reduce_(x, bits, nearest):
    if bits >= 52:
        return
    drop = 52 - bits
    _RED64(np.uint64((1 << (drop - 1)) if nearest else 0), np.uint64(((1 << 64) - 1) ^ ((1 << drop) - 1)),
           x.view(cp.uint64).T)


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
        Lt = A[j + b:, j:j + b].T.copy().T  # column-major copy, read in memory order (see _RED64)
        if not fp32:
            reduce_(Lt, bits, nearest)
            if stats and j % (8 * NB) == 0:
                span_stats(Lt)
        assert gemm(h, 0, 1, r, r, b, ctypes.byref(mone), ctypes.c_void_p(Lt.data.ptr), r,
                    ctypes.c_void_p(Lt.data.ptr), r, ctypes.byref(one), ptr(j + b, j + b), n) == 0
    cp.cuda.Device().synchronize()


FAR_UPDATE = cp.RawKernel(r"""
extern "C" __global__ void far_update(double* A, const float* T, long n, long r0, long m, long nc, long ldt, long nb,
                                       const int* bend) {
  // A points at row r0 of the trailing block's first column inside the column-major n x n matrix; T holds the
  // FP16-input product of trailing rows r0..r0+m-1 with columns 0..nc-1. Tiles below the FP64 band of their
  // column (tile(r0 + i) > bend[tile(c)]) take it: A[i, c] -= T[i, c].
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= m * nc) return;
  long c = idx / m, i = idx - c * m;
  if ((r0 + i) / nb <= bend[c / nb]) return;
  A[i + c * n] -= (double)T[i + c * ldt];
}
""", "far_update")


def reduce16_(x, bits):
    if bits >= 10:
        return
    drop = 10 - bits
    _RED16(np.uint16(1 << (drop - 1)), np.uint16(0xFFFF ^ ((1 << drop) - 1)), x.view(cp.uint16).T)


def fp64_extent(A, mode):
    """Last FP64 tile row of every tile column. mode = band D (tiles within D of the diagonal stay FP64), or 'n'
    for the tile-norm rule of the ExaGeoStat line (Higham and Mary): tile (I, J) may use FP16 when
    ||A_IJ||_F < u_req ||A||_F / (nt u_fp16), u_req = GP_MP_TOL (1e-9), u_fp16 = 2^-11; the FP64 extent of a
    column then reaches its last tile above that threshold (lower-norm tiles inside it stay FP64)."""
    n = A.shape[0]
    nt = -(-n // NB)
    if mode != "n":
        ext = np.minimum(np.arange(nt) + int(mode) - 1, nt - 1)
        return ext, sum(nt - 1 - ext[J] for J in range(nt)) / (nt * (nt + 1) / 2)
    nrm = cp.zeros((nt, nt))
    for J in range(nt):
        sq = (A[:, J * NB:(J + 1) * NB] ** 2).sum(axis=1)
        sq = cp.concatenate([sq, cp.zeros(nt * NB - n)])
        nrm[:, J] = cp.sqrt(sq.reshape(nt, NB).sum(axis=1))
    nrm = cp.asnumpy(nrm)
    thr = float(os.environ.get("GP_MP_TOL", 1e-9)) * math.sqrt((nrm ** 2).sum()) / (nt * 2.0 ** -11)
    low = np.tril(nrm < thr, -1)
    ext = np.array([max([J] + [I for I in range(J, nt) if not low[I, J]]) for J in range(nt)])
    frac16 = sum(nt - 1 - ext[J] for J in range(nt)) / (nt * (nt + 1) / 2)
    return ext, frac16


MP_FRAC = []


def cholesky_mp(A, mode, bits, nearest, hbits):
    """Mixed-precision version of cholesky() in the style of ExaGeoStat's tile methods. In the trailing update
    A22 -= L21 L21^T the tiles of each column down to its FP64 extent (fp64_extent) are computed in FP64 (one DGEMM
    per tile column, operand optionally mantissa-reduced, +t); the tiles below come from FP16 copies of L21 with
    FP32 accumulation on the tensor cores (one GEMM over the trailing block per panel; FP16 mantissa optionally
    reduced, +h) and are subtracted from the FP64 matrix."""
    n = A.shape[0]
    ext, frac16 = fp64_extent(A, mode)
    MP_FRAC.append(frac16)
    h, L = handle(), lib()
    one, mone = ctypes.c_double(1.0), ctypes.c_double(-1.0)
    fone, fzero = ctypes.c_float(1.0), ctypes.c_float(0.0)
    base = A.data.ptr
    ptr = lambda i, j: ctypes.c_void_p(base + (i + j * n) * 8)  # noqa: E731
    ldt = max(n - NB, 1)
    T = None
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
        assert L.cublasDtrsm_v2(h, 1, 0, 1, 0, r, b, ctypes.byref(one), ptr(j, j), n, ptr(j + b, j), n) == 0
        Lt = A[j + b:, j:j + b].T.copy().T  # column-major copy, read in memory order
        reduce_(Lt, bits, nearest)
        o, p1 = j + b, j // NB + 1        # trailing block starts at (o, o) = tile p1
        R = -(-r // NB)
        bend = np.array([ext[p1 + c] - p1 for c in range(R)], dtype=np.int32)  # local last FP64 tile row
        for c in range(R):                # FP64 band of each tile column
            r0, c1 = c * NB, min((c + 1) * NB, r)
            r1 = min(int(bend[c] + 1) * NB, r)  # plain ints: ctypes rejects numpy integers
            assert L.cublasDgemm_v2(h, 0, 1, r1 - r0, c1 - r0, b, ctypes.byref(mone),
                                    ctypes.c_void_p(Lt.data.ptr + r0 * 8), r, ctypes.c_void_p(Lt.data.ptr + r0 * 8), r,
                                    ctypes.byref(one), ptr(o + r0, o + r0), n) == 0
        far = np.nonzero(bend < R - 1)[0]
        if far.size:                      # tiles below the band: FP16 inputs, FP32 accumulation, computed only
            r0 = int(bend[far].min() + 1) * NB                  # over the rows and columns that hold them
            m, nc = r - r0, min(int(far.max() + 1) * NB, r)
            if T is None:
                T = cp.empty((ldt, ldt), dtype=cp.float32, order='F')
            Lh = Lt.T.astype(cp.float16).T
            reduce16_(Lh, hbits)
            assert L.cublasGemmEx(h, 0, 1, m, nc, b, ctypes.byref(fone), ctypes.c_void_p(Lh.data.ptr + r0 * 2), 2, r,
                                  ctypes.c_void_p(Lh.data.ptr), 2, r, ctypes.byref(fzero), ctypes.c_void_p(T.data.ptr),
                                  0, ldt, 68, -1) == 0
            FAR_UPDATE((-(-m * nc // 256),), (256,),
                       (A[o + r0:, o:], T, cp.int64(n), cp.int64(r0), cp.int64(m), cp.int64(nc), cp.int64(ldt),
                        cp.int64(NB), cp.asarray(bend)))
            del Lh
    cp.cuda.Device().synchronize()
    del T


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


def factor(A, bits, nearest, fp32, mp=None, stats=False):
    if mp is None:
        cholesky(A, bits, nearest, fp32, stats=stats)
    else:
        cholesky_mp(A, mp[0], bits, nearest, mp[1])


def eval_ll(rng, bits, nearest, fp32, mp=None):
    """One log-likelihood evaluation at range parameter rng (NaN if the factorisation fails)."""
    A = covariance(xy, rng, cp.float32 if fp32 else cp.float64)
    try:
        factor(A, bits, nearest, fp32, mp)
        ll = loglik(A, y)
    except Exception:  # noqa: BLE001
        ll = float("nan")
    del A
    cp.get_default_memory_pool().free_all_blocks()
    return ll


MLE_REF = {}


def mle_run(v, pas, bits, nearest, fp32, mp=None):
    """Maximum-likelihood estimate of the range parameter by golden-section search on log10(range):
    the end-to-end task of a geostatistics fit. Reports the estimate, the number of likelihood evaluations
    and the wall time, so variants are compared on the answer a user would get, not only on one value."""
    lo, hi = math.log10(float(os.environ.get("GP_LO", 0.01))), math.log10(float(os.environ.get("GP_HI", 0.2)))
    tol = float(os.environ.get("GP_TOL", 0.002))
    g = (math.sqrt(5) - 1) / 2
    f = lambda x: eval_ll(10 ** x, bits, nearest, fp32, mp)  # noqa: E731
    t0 = time.time()
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    nev = 2
    while b - a > tol:
        if (fc if fc == fc else -math.inf) >= (fd if fd == fd else -math.inf):
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        nev += 1
    xh = (a + b) / 2
    llh = f(xh)
    nev += 1
    t1 = time.time()
    th = 10 ** xh
    if v == "native" and "theta" not in MLE_REF:
        MLE_REF.update(theta=th, ll=llh)
    dth = (th / MLE_REF["theta"] - 1) if MLE_REF else float("nan")
    dll = (llh - MLE_REF["ll"]) if MLE_REF else float("nan")
    print(f"RESULT_MLE tdp={os.environ.get('GP_CAP', '-')} variant={v} pass={pas} theta={th:.8f} loglik={llh:.6f} "
          f"dtheta_rel={dth:.3e} dll={dll:.3e} evals={nev} wall={t1 - t0:.3f}s t_begin={t0:.3f} t_end={t1:.3f}",
          flush=True)


def morton_order(p):
    """Indices that sort 2D points in [0,1)^2 along a Morton (Z-order) curve, 16 bits per coordinate."""
    q = np.minimum((p * 65536).astype(np.uint64), 65535)

    def spread(v):
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
        return (v | (v << np.uint64(1))) & np.uint64(0x55555555)

    return np.argsort(spread(q[:, 0]) | (spread(q[:, 1]) << np.uint64(1)), kind="stable")


rs = np.random.default_rng(int(os.environ.get("GP_SEED", 1)))
xy_h = rs.random((N, 2))
if os.environ.get("GP_ORDER") == "morton":
    xy_h = xy_h[morton_order(xy_h)]
xy = cp.asarray(xy_h)
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
print(f"n={N} nb={NB} ranges={RANGES} nugget={NUGGET} order={os.environ.get('GP_ORDER', 'none')}", flush=True)

ref = None
seen = {}
for v in os.environ.get("GP_VARIANTS", "native").split():
    seen[v] = seen.get(v, 0) + 1
    pas = "ABCDEFGH"[seen[v] - 1]
    bits, nearest, fp32, emu, mp = 52, True, False, None, None
    if v.startswith("mp"):
        parts = v.split("+")
        mp = [parts[0][2:], 10]
        for q in parts[1:]:
            if q.startswith("t"):
                bits, nearest = int(q[1:-1]), q[-1] == "n"
            elif q.startswith("h"):
                mp[1] = int(q[1:])
    elif v.startswith("t"):
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
    if os.environ.get("GP_MODE") == "mle":
        mle_run(v, pas, bits, nearest, fp32, mp)
        set_emulation(None)
        continue
    t0 = time.time()
    lls = []
    status = "ok"
    for rng in RANGES:
        A = covariance(xy, rng, cp.float32 if fp32 else cp.float64)
        try:
            factor(A, bits, nearest, fp32, mp, stats=(v.startswith("t") or v == "native") and seen[v] == 1)
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
    if MP_FRAC:
        print("MP_GP frac_fp16=[" + " ".join(f"{x:.3f}" for x in MP_FRAC) + "]", flush=True)
        MP_FRAC.clear()
    if SPAN:
        print("RANGE_GP " + " ".join(f"frac_below{t}={a / max(b, 1):.3e}" for t, (a, b) in sorted(SPAN.items())), flush=True)
        SPAN.clear()
