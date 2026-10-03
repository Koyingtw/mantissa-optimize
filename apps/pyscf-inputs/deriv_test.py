# 用法: python deriv_test.py <xyz> <grad|hess|tda> [xc] [basis]
# DF-RKS 之後做能量梯度、Hessian 或 TDA 激發態；這些都大量使用格點上的原子軌域（及其導數）矩陣乘法。
import os
import sys
import time

import cupy
import numpy as np
from pyscf import gto
from gpu4pyscf import dft

xyz, task = sys.argv[1], sys.argv[2]
xc = sys.argv[3] if len(sys.argv) > 3 else 'b3lyp'
basis = sys.argv[4] if len(sys.argv) > 4 else 'def2-tzvp'
mol = gto.M(atom=xyz, basis=basis, verbose=4, max_memory=200000)
print(f"natm={mol.natm} nao={mol.nao} task={task} xc={xc} basis={basis}")
mf = dft.RKS(mol, xc=xc).density_fit()
mf.grids.level = int(os.environ.get("GRID_LEVEL", "3"))
mf.conv_tol = float(os.environ.get("CONV_TOL", "1e-10"))
t0 = time.perf_counter()
e = mf.kernel()
cupy.cuda.Device().synchronize()
t1 = time.perf_counter()
if task == 'grad':
    out = np.asarray(mf.nuc_grad_method().kernel())
    val = f"grad_norm={np.linalg.norm(out):.10e}"
elif task == 'hess':
    out = np.asarray(mf.Hessian().kernel())
    val = f"hess_norm={np.linalg.norm(out):.10e}"
else:
    td = mf.TDA()
    td.nstates = 10
    out = cupy.asnumpy(td.kernel()[0]) * 27.211386
    val = "exc_eV=[" + " ".join(f"{x:.6f}" for x in out) + "]"
cupy.cuda.Device().synchronize()
t2 = time.perf_counter()
print(f"RESULT converged={mf.converged} E={e:.10f} Eh  scf_wall={t1 - t0:.2f}s  {task}_wall={t2 - t1:.2f}s  wall={t2 - t0:.2f}s  {val}")
