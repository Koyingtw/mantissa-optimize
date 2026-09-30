"""Conjugate gradient on a sparse SPD matrix whose stored FP64 values keep only --frac mantissa bits.

The sparse matrix-vector product (cuSPARSE CSR SpMV through PyTorch) is memory bound, so this is
the memory-bound FP64 counterpart of the GEMM experiments: storage stays FP64, only the low
mantissa bits of the matrix values are zeroed (round to nearest).  The right-hand side is
b = A_orig @ ones, so the exact solution is all ones.  Jacobi-preconditioned CG runs to --tol on
the reduced matrix; repeated solves fill the --seconds window.  RESULT line as gemm_mant_bench
(dtype=cg): `tflops` = SpMV GFLOP/s / 1000, plus iters per solve, the true residual
||b - A_orig x|| / ||b|| and the max error of x.
Matrix: --mtx FILE (MatrixMarket, SPD) or a generated 3-D 7-point variable-coefficient diffusion
operator on an --grid^3 mesh with log-normal coefficients (full-entropy values).
"""
import argparse
import math
import time

import torch


def reduce64(v, k):
    if k >= 52:
        return v
    u = v.view(torch.int64)
    drop = 52 - k
    u = u + ((1 << (drop - 1)) - 1) + ((u >> drop) & 1)
    u = u & ~((1 << drop) - 1)
    return u.view(torch.float64)


def diffusion3d(n, seed=0):
    """-div(c grad u) on an n^3 grid, 7-point stencil, c log-normal per cell face (SPD)."""
    g = torch.Generator().manual_seed(seed)
    N = n ** 3
    idx = torch.arange(N).view(n, n, n)
    rows, cols, vals = [], [], []
    diag = torch.zeros(N, dtype=torch.float64)
    for dim in range(3):
        a = idx.narrow(dim, 0, n - 1).reshape(-1)
        b = idx.narrow(dim, 1, n - 1).reshape(-1)
        c = torch.exp(torch.randn(a.numel(), generator=g, dtype=torch.float64))  # face coefficient
        rows += [a, b]
        cols += [b, a]
        vals += [-c, -c]
        diag.index_add_(0, a, c)
        diag.index_add_(0, b, c)
    diag += 1e-2  # Dirichlet-like shift keeps it well posed
    rows.append(torch.arange(N))
    cols.append(torch.arange(N))
    vals.append(diag)
    r, c, v = torch.cat(rows), torch.cat(cols), torch.cat(vals)
    return torch.sparse_coo_tensor(torch.stack([r, c]), v, (N, N)).coalesce().to_sparse_csr()


def load_mtx(path):
    import scipy.io
    import scipy.sparse
    m = scipy.sparse.csr_matrix(scipy.io.mmread(path))
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr).long(), torch.from_numpy(m.indices).long(),
                                   torch.from_numpy(m.data).double(), m.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="cg")
    ap.add_argument("--frac", type=int, default=52)
    ap.add_argument("--mtx", default="")
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--maxit", type=int, default=5000)
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--low", default="zero")
    a, _ = ap.parse_known_args()

    import os
    cache = a.mtx + ".pt" if a.mtx else f"/nfs/home/koying/models/cg_diffusion{a.grid}.pt"
    if os.path.exists(cache):
        crow, col, val0, shape = torch.load(cache)
    else:
        m = load_mtx(a.mtx) if a.mtx else diffusion3d(a.grid)
        crow, col, val0, shape = m.crow_indices().int(), m.col_indices().int(), m.values(), m.shape
        torch.save((crow, col, val0, shape), cache)
    # 32-bit indices, as most applications store them (cuSPARSE CSR SpMV)
    crow, col, val0 = crow.int().cuda(), col.int().cuda(), val0.cuda()
    A0 = torch.sparse_csr_tensor(crow, col, val0, shape)
    A = torch.sparse_csr_tensor(crow, col, reduce64(val0.clone(), a.frac), shape)
    n, nnz = A0.shape[0], val0.numel()
    ones = torch.ones(n, dtype=torch.float64, device="cuda")
    b = A0 @ ones
    rows = torch.repeat_interleave(torch.arange(n, device="cuda"), (crow[1:] - crow[:-1]).long())
    dinv = 1.0 / torch.zeros(n, dtype=torch.float64, device="cuda").index_add_(
        0, rows, torch.where(rows == col.long(), A.values(), torch.zeros_like(A.values())))

    def solve():
        x = torch.zeros_like(b)
        r = b.clone()
        z = dinv * r
        p = z.clone()
        rz = torch.dot(r, z)
        bn = torch.linalg.norm(b)
        it = 0
        while it < a.maxit:
            q = A @ p
            alpha = rz / torch.dot(p, q)
            x += alpha * p
            r -= alpha * q
            it += 1
            if it % 10 == 0 and (torch.linalg.norm(r) / bn).item() < a.tol:
                break
            z = dinv * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return x, it

    t = time.time()
    while time.time() - t < a.warmup:
        solve()
    torch.cuda.synchronize()
    iters_total, solves = 0, 0
    t_begin = time.time()
    while time.time() - t_begin < a.seconds:
        x, it = solve()
        iters_total += it
        solves += 1
    torch.cuda.synchronize()
    t_end = time.time()
    true_res = (torch.linalg.norm(b - A0 @ x) / torch.linalg.norm(b)).item()
    err = (x - 1).abs().max().item()
    gflops = 2.0 * nnz * iters_total / (t_end - t_begin) / 1e9
    print(f"RESULT tag={a.tag} dtype=cg frac={a.frac} only_a=0 low={a.low} round=nearest m={n} n={nnz} "
          f"k={iters_total // max(solves, 1)} calls={solves} tflops={gflops / 1000:.5f} chunk_min=0 chunk_max=0 "
          f"t_begin={t_begin:.3f} t_end={t_end:.3f} rel_err={true_res:.3e} emu_bits=-1 xerr={err:.3e}", flush=True)


if __name__ == "__main__":
    main()
