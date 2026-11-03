"""纈胺黴素 DFT 的真實 GEMM 運算元（CPU，PySCF；第八輪審查 Q1）：GH200 上 GPU4PySCF 的 INT8 模擬在這個分子上「安靜地」
算錯（cuBLAS 模擬 36 位差 52.5 mEh，GEMMul8 發散），GPU4PySCF 不支援 ROCm，所以在 CPU 上做同一個計算（DF-B3LYP／def2-TZVP，
與 GH200 相同的分子與基底），取出 GPU4PySCF 交換相關（XC）部分的三種 GEMM 運算元，給 apps/g8_spot.hip 在 MI350X 上檢驗
事後抽查：
  xc4096   格點區塊（4096 點，最靠近分子中心）上「有意義」的 AO（|值| 最大 > 1e-15）：C = ao^T (w v ao)，
           m = n = 區塊內 AO 數，k = 4096（GH200 上逐列跨度最寬的形狀，約 1390×1390×4096）
  xc76800  同上，76800 點、全部 AO：m = n = 2958，k = 76800（長 k 形狀）
  mo131072 131072 個格點上的 AO 乘佔據軌域係數：C = ao C_occ，m = 131072，n = nocc = 300，k = 2958（瘦長形狀）
w 是積分權重、v 是收斂密度的 LDA 位勢（只決定權重的大小分布）。每組寫 A.bin、B.bin（N N 排列，column-major：A 為 m×k、
B 為 k×n；mo131072 的 A 改存 k×m，用 --ta）與 meta.txt。
usage: valino_prep.py <xyz> <outdir>
"""
import os
import sys
import time

import numpy as np
from pyscf import dft, gto, lib

xyz, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
lib.num_threads(int(os.environ.get("OMP_NUM_THREADS", 128)))
t0 = time.time()
mol = gto.M(atom=xyz, basis="def2-tzvp", verbose=4, max_memory=1500000)
print(f"natm={mol.natm} nao={mol.nao} nelec={mol.nelectron}", flush=True)
mf = dft.RKS(mol, xc="B3LYP").density_fit()
mf.conv_tol = 1e-7
e = mf.kernel()
print(f"E(B3LYP)={e:.10f} converged={mf.converged} scf_time={time.time() - t0:.1f}s", flush=True)

g = mf.grids
center = mol.atom_coords().mean(axis=0)
order = np.argsort(np.linalg.norm(g.coords - center, axis=1), kind="stable")
nocc = mol.nelectron // 2
cocc = mf.mo_coeff[:, :nocc]
dm = mf.make_rdm1()


def weighted(coords, weights, ao):
    rho = np.einsum("gi,ij,gj->g", ao, dm, ao, optimize=True)
    vrho = dft.libxc.eval_xc("LDA,VWN", rho, deriv=1)[1][0]
    return ao * (weights * vrho)[:, None]


def save(name, A, B, meta):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(A, dtype=np.float64).tofile(os.path.join(d, "A.bin"))
    np.ascontiguousarray(B, dtype=np.float64).tofile(os.path.join(d, "B.bin"))
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(meta + "\n")
    print(name, meta, flush=True)


for npts, sig in ((4096, True), (76800, False)):
    sel = np.sort(order[:npts])  # a compact block of grid points, kept in PySCF's order
    ao = dft.numint.eval_ao(mol, g.coords[sel])  # (npts, nao)
    if sig:
        keep = np.abs(ao).max(axis=0) > 1e-15
        ao = ao[:, keep]
        aow = weighted(g.coords[sel], g.weights[sel], dft.numint.eval_ao(mol, g.coords[sel]))[:, keep]
    else:
        aow = weighted(g.coords[sel], g.weights[sel], ao)
    nsig = ao.shape[1]
    # op(A) = ao^T (m = nsig x k = npts, column-major) is ao in C order; op(B) = aow (k x n) is aow^T in C order
    save(f"xc{npts}", ao, aow.T, f"m={nsig} n={nsig} k={npts} layout=NN")

sel = np.sort(order[:131072])
ao = dft.numint.eval_ao(mol, g.coords[sel])
# op(A) = ao (m = grid x k = nao) with --ta: stored k x m column-major = ao in C order; B = C_occ (k x n) = C_occ^T in C order
save("mo131072", ao, cocc.T, f"m={ao.shape[0]} n={nocc} k={mol.nao} layout=TN")
print(f"done {time.time() - t0:.1f}s", flush=True)
