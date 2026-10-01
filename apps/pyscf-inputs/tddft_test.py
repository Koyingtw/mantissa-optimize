# 用法: python tddft_test.py <xyz> [xc] [basis] [nstates]
# GPU DF-RKS 之後做 TDA（Tamm-Dancoff）激發態；數值積分（格點上的原子軌域矩陣乘法）佔很大比例。
import os
import sys
import time

import cupy
from pyscf import gto
from gpu4pyscf import dft

xyz = sys.argv[1]
xc = sys.argv[2] if len(sys.argv) > 2 else 'b3lyp'
basis = sys.argv[3] if len(sys.argv) > 3 else 'def2-tzvp'
nstates = int(sys.argv[4]) if len(sys.argv) > 4 else 10
mol = gto.M(atom=xyz, basis=basis, verbose=4, max_memory=200000)
print(f"natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} xc={xc} basis={basis} nstates={nstates}")
mf = dft.RKS(mol, xc=xc).density_fit()
mf.grids.level = int(os.environ.get("GRID_LEVEL", "3"))
mf.conv_tol = float(os.environ.get("CONV_TOL", "1e-10"))
t0 = time.perf_counter()
e = mf.kernel()
cupy.cuda.Device().synchronize()
t1 = time.perf_counter()
td = mf.TDA()
td.nstates = nstates
td.conv_tol = float(os.environ.get("TD_CONV_TOL", "1e-5"))
exc = td.kernel()[0]
cupy.cuda.Device().synchronize()
t2 = time.perf_counter()
ev = " ".join(f"{x * 27.211386:.6f}" for x in cupy.asnumpy(exc))
print(f"RESULT converged={mf.converged} E={e:.10f} Eh  scf_wall={t1 - t0:.2f}s  td_converged={all(td.converged)} "
      f"td_wall={t2 - t1:.2f}s  wall={t2 - t0:.2f}s  exc_eV=[{ev}]")
