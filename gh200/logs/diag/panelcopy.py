# Diagnostic: panel copy and mantissa reduction on column-major data, index-order vs memory-order traversal.
import time
import cupy as cp
import numpy as np
n, b, j = 60000, 1024, 1024
A = cp.asfortranarray(cp.random.standard_normal((n, n // 4)))  # 60000 x 15000 column-major stand-in
def timed(f, k=5):
    f(); cp.cuda.Device().synchronize(); t = time.time()
    for _ in range(k): f()
    cp.cuda.Device().synchronize(); return 1e3 * (time.time() - t) / k
R64 = cp.ElementwiseKernel('uint64 add, uint64 mask', 'uint64 u', 'u = (u + add) & mask', 'r64')
R16 = cp.ElementwiseKernel('uint16 add, uint16 mask', 'uint16 u', 'u = (u + add) & mask', 'r16')
print("copy asfortranarray   %.3f ms" % timed(lambda: cp.asfortranarray(A[j + b:, j:j + b])))
print("copy .T.copy().T      %.3f ms" % timed(lambda: A[j + b:, j:j + b].T.copy().T))
Lt = A[j + b:, j:j + b].T.copy().T
add, mask = np.uint64(1 << 23), np.uint64(((1 << 64) - 1) ^ ((1 << 24) - 1))
print("red64 index order     %.3f ms" % timed(lambda: R64(add, mask, Lt.view(cp.uint64))))
print("red64 memory order    %.3f ms" % timed(lambda: R64(add, mask, Lt.view(cp.uint64).T)))
print("astype index order    %.3f ms" % timed(lambda: Lt.astype(cp.float16)))
print("astype memory order   %.3f ms" % timed(lambda: Lt.T.astype(cp.float16).T))
Lh = Lt.T.astype(cp.float16).T
print("red16 memory order    %.3f ms" % timed(lambda: R16(np.uint16(16), np.uint16(0xFFE0), Lh.view(cp.uint16).T)))
x1 = Lt.copy(order='F'); x2 = Lt.copy(order='F')
R64(add, mask, x1.view(cp.uint64)); R64(add, mask, x2.view(cp.uint64).T)
print("same result:", bool((x1 == x2).all()), " Lh F-contig:", Lh.flags.f_contiguous, " Lt F-contig:", Lt.flags.f_contiguous)
