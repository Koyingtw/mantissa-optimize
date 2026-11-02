"""Flip statistics of the GEMM bench's uniform operands, recomputed on the CPU (numpy re-implementation of
bench/gemm_mant_bench.hip's generator: splitmix64 hash, uniform in [-1,1), low fraction bits truncated or rounded,
zero fill) on a sample of rows. Gives, per kept-bit count, the mean bit flips between neighbouring operands along k
(tog) and across rows/columns (tox), averaged over A and B, as `--stats-only` would.
usage (module): flips(frac, m=32768, n=32768, k=2048, rows=2048, rnd=False) -> (tog, tox)
"""
import numpy as np

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & M64
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & M64
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & M64
    return x ^ (x >> np.uint64(31))


def operand_bits(seed, idx, frac, rnd=False):
    with np.errstate(over="ignore"):
        z = splitmix64(np.uint64(seed) ^ (idx.astype(np.uint64) * np.uint64(0xD1B54A32D192ED03)))
    v = (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -52 - 1.0
    u = v.view(np.uint64)
    if frac < 52:
        drop = 52 - frac
        if rnd:
            u = u + np.uint64(1 << (drop - 1))
        u = u & ~np.uint64((1 << drop) - 1)
    return u


def popcount(x):
    return np.unpackbits(x.view(np.uint8), axis=-1).reshape(*x.shape, 64).sum(-1)


def flips(frac, m=32768, n=32768, k=2048, rows=1024, rnd=False):
    r = np.arange(rows, dtype=np.uint64)[:, None]
    j = np.arange(k, dtype=np.uint64)[None, :]
    A = operand_bits(1, j * np.uint64(m) + r, frac, rnd)          # A: m x k col-major, rows r, all k
    B = operand_bits(2, r * np.uint64(k) + j, frac, rnd)          # B: k x n col-major, columns r, all k
    tog = 0.5 * (popcount(A[:, 1:] ^ A[:, :-1]).mean() + popcount(B[:, 1:] ^ B[:, :-1]).mean())
    tox = 0.5 * (popcount(A[1:, :] ^ A[:-1, :]).mean() + popcount(B[1:, :] ^ B[:-1, :]).mean())
    return tog, tox


if __name__ == "__main__":
    for f in (52, 27, 0):
        print(f, [round(x, 3) for x in flips(f)])
