# Diagnostic: cost of FP16 operand mantissa reduction and of the FP16 GEMM with reduced vs full operands.
import ctypes, time
import cupy as cp

def lib():
    cp.cuda.device.get_cublas_handle()
    path = next(l.split()[-1] for l in open('/proc/self/maps') if '/libcublas.so' in l)
    return ctypes.CDLL(path)

L = lib(); h = ctypes.c_void_p(cp.cuda.device.get_cublas_handle())
r, b = 58000, 1024
Lt = cp.asfortranarray(cp.random.standard_normal((r, b)) * 0.05)
T = cp.empty((r, r), dtype=cp.float32, order='F')
one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)

def reduce16_(x, bits):
    u = x.view(cp.uint16)
    drop = 10 - bits
    u += cp.uint16(1 << (drop - 1))
    u &= cp.uint16(0xFFFF ^ ((1 << drop) - 1))

def timed(f, n=5):
    f(); cp.cuda.Device().synchronize()
    t = time.time()
    for _ in range(n):
        f()
    cp.cuda.Device().synchronize()
    return (time.time() - t) / n

def conv(bits):
    Lh = Lt.astype(cp.float16)
    if bits < 10:
        reduce16_(Lh, bits)
    return Lh

Lh10, Lh5 = conv(10), conv(5)
gemm = lambda Lh: L.cublasGemmEx(h, 0, 1, r, r, b, ctypes.byref(one), ctypes.c_void_p(Lh.data.ptr), 2, r,
                                 ctypes.c_void_p(Lh.data.ptr), 2, r, ctypes.byref(zero), ctypes.c_void_p(T.data.ptr), 0, r, 68, -1)
print("astype only      %.3f ms" % (1e3 * timed(lambda: conv(10))))
print("astype+reduce16  %.3f ms" % (1e3 * timed(lambda: conv(5))))
for name, Lh in (("fp16 full", Lh10), ("fp16 5 bits", Lh5)):
    print("%-12s gemm %.2f ms" % (name, 1e3 * timed(lambda: gemm(Lh))))
print("subnormal/zero fraction 5 bits:", float((cp.abs(Lh5) < 6.1e-5).mean()), " full:", float((cp.abs(Lh10) < 6.1e-5).mean()))

# fused single-pass versions (as now used in gp_mle.py)
import numpy as np
R16 = cp.ElementwiseKernel('uint16 add, uint16 mask', 'uint16 u', 'u = (u + add) & mask', 'r16')
R64 = cp.ElementwiseKernel('uint64 add, uint64 mask', 'uint64 u', 'u = (u + add) & mask', 'r64')
def conv_f(bits):
    Lh = Lt.astype(cp.float16)
    R16(np.uint16(16), np.uint16(0xFFE0), Lh.view(cp.uint16))
    return Lh
print("astype+fused16   %.3f ms" % (1e3 * timed(lambda: conv_f(5))))
X = Lt.copy(order='F')
def old64():
    u = X.view(cp.uint64); u += cp.uint64(1 << 23); u &= cp.uint64(((1 << 64) - 1) ^ ((1 << 24) - 1))
print("fp64 two-op      %.3f ms" % (1e3 * timed(old64)))
print("fp64 fused       %.3f ms" % (1e3 * timed(lambda: R64(np.uint64(1 << 23), np.uint64(((1 << 64) - 1) ^ ((1 << 24) - 1)), X.view(cp.uint64)))))
print("fp64 copy (asfortranarray of a strided block) %.3f ms" % (1e3 * timed(lambda: cp.asfortranarray(X[:, :]))))
y1 = Lt.copy(order='F'); y2 = Lt.copy(order='F')
u = y1.view(cp.uint64); u += cp.uint64(1 << 23); u &= cp.uint64(((1 << 64) - 1) ^ ((1 << 24) - 1))
R64(np.uint64(1 << 23), np.uint64(((1 << 64) - 1) ^ ((1 << 24) - 1)), y2.view(cp.uint64))
print("fused == two-op:", bool((y1 == y2).all()))
