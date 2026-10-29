"""DF-MP2 前處理（CPU，PySCF）：DF-RHF → MO 基底的三中心張量 B[i][a][P] 與軌域能量，給 mi350/apps/dfmp2.hip 用。

GPU4PySCF 不支援 ROCm，所以 MI350 上把 DF-MP2 拆成兩段：這裡在 CPU 上做 SCF 與積分轉換（與 GPU4PySCF 相同的
def2-TZVP／RI 輔助基底），GPU 端只做佔總時間絕大部分的收縮 g_ab = B_i B_j^T（FP64 DGEMM，內維度 = 輔助基底數）
與能量求和。同時用 PySCF 的 CPU DF-MP2 算參考相關能，驗證 GPU 實作。

usage: mp2_prep.py <xyz> <outdir> [basis=def2-tzvp]
輸出：<outdir>/B.bin（float64，nocc × nvir × naux，C order）、eps.bin（float64，nmo）、meta.txt（nocc nvir naux 與參考能量）
"""
import os
import sys
import time

import numpy as np
from pyscf import df, gto, lib, scf
from pyscf.mp import dfmp2

xyz, out = sys.argv[1], sys.argv[2]
basis = sys.argv[3] if len(sys.argv) > 3 else "def2-tzvp"
os.makedirs(out, exist_ok=True)
lib.num_threads(int(os.environ.get("OMP_NUM_THREADS", 128)))
t0 = time.time()
mol = gto.M(atom=xyz, basis=basis, verbose=4, max_memory=1500000)
print(f"natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} basis={basis}", flush=True)
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10
e_hf = mf.kernel()
print(f"E(HF)={e_hf:.12f} scf_time={time.time() - t0:.1f}s", flush=True)

mp = dfmp2.DFRMP2(mf)
# PySCF 2.x 的 DFRMP2 預設沿用 SCF 的 JK-fit 輔助基底；MP2 標準用 RI 基底（make_auxbasis(mp2fit=True)，
# def2-TZVP 即 def2-TZVP-RI），這裡明確指定
mp.with_df = df.DF(mol, auxbasis=df.make_auxbasis(mol, mp2fit=True))
nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
t1 = time.time()
eris = dfmp2._make_df_eris(mp)          # 不凍結核心：occ = 全部佔據軌域
B = np.asarray(eris.ovL)                # (nocc*nvir, naux)，C order = B[i][a][P]
naux = B.shape[1]
assert B.shape[0] == nocc * nvir
print(f"nocc={nocc} nvir={nvir} naux={naux} B_GiB={B.nbytes / 2**30:.1f} ao2mo_time={time.time() - t1:.1f}s", flush=True)
mf.mo_energy.astype(np.float64).tofile(f"{out}/eps.bin")
B.tofile(f"{out}/B.bin")

t2 = time.time()
e_corr = dfmp2.kernel(mp, eris=eris, with_t2=False)[0]
print(f"E_corr(PySCF CPU DF-MP2)={e_corr:.12f} mp2_time={time.time() - t2:.1f}s", flush=True)
with open(f"{out}/meta.txt", "w") as f:
    f.write(f"nocc={nocc} nvir={nvir} naux={naux} e_hf={e_hf:.12f} e_corr_ref={e_corr:.12f} basis={basis} xyz={xyz}\n")
