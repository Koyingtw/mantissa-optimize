# 用法: python mp2_test.py <xyz> [basis]   (GPU: DF-RHF + DF-MP2，DF-MP2 幾乎全是 DGEMM)
import sys, time, os
import cupy
from pyscf import gto
from gpu4pyscf import scf
from gpu4pyscf.mp import dfmp2
import gpu4pyscf.mp.dfmp2_addons as _A
# cupy>=14 沒有 cuTENSOR → contract 走 cupy.einsum（→ cuBLAS），會多出暫存陣列；
# 原本 j3c batch 以 80% 可用記憶體估算會 OOM，這裡降為 MP2_MEM_PREFACTOR（預設 0.3）
_orig_est = _A.estimate_j3c_batch
_A.estimate_j3c_batch = lambda *a, **k: _orig_est(*a, **{**k, 'prefactor': float(os.environ.get('MP2_MEM_PREFACTOR', 0.3))})
basis = sys.argv[2] if len(sys.argv) > 2 else 'def2-tzvp'
mol = gto.M(atom=sys.argv[1], basis=basis, verbose=4, max_memory=200000)
print(f"natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} basis={basis}")
mf = scf.RHF(mol).density_fit(); mf.conv_tol = 1e-10
t0 = time.perf_counter(); ehf = mf.kernel(); cupy.cuda.Device().synchronize(); t1 = time.perf_counter()
print(f"RESULT_HF converged={mf.converged} E={ehf:.10f} Eh wall={t1-t0:.2f}s")
pt = dfmp2.DFMP2(mf)
t2 = time.perf_counter(); ecorr, _ = pt.kernel(); cupy.cuda.Device().synchronize(); t3 = time.perf_counter()
print(f"RESULT_MP2 Ecorr={float(ecorr):.10f} Etot={float(ehf+ecorr):.10f} Eh wall={t3-t2:.2f}s")
