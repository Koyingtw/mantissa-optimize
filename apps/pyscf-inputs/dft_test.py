# 用法: python dft_test.py <xyz> <gpu|cpu> [xc] [basis] [df|direct]
import sys, time, os
import pyscf
from pyscf import gto
xyz = sys.argv[1]; dev = sys.argv[2]
xc = sys.argv[3] if len(sys.argv) > 3 else 'b3lyp'
basis = sys.argv[4] if len(sys.argv) > 4 else 'def2-tzvp'
mode = sys.argv[5] if len(sys.argv) > 5 else 'df'
mol = gto.M(atom=xyz, basis=basis, verbose=4, max_memory=200000)
print(f"natm={mol.natm} nao={mol.nao} nelec={mol.nelectron} xc={xc} basis={basis} mode={mode} dev={dev}")
if dev == 'gpu':
    import cupy, gpu4pyscf
    from gpu4pyscf import dft
    print("gpu4pyscf", gpu4pyscf.__version__, "cupy", cupy.__version__, "CONTRACT_ENGINE", os.environ.get('CONTRACT_ENGINE'))
else:
    from pyscf import dft
mf = dft.RKS(mol, xc=xc)
if mode == 'df':
    mf = mf.density_fit()
mf.grids.level = 3
mf.conv_tol = float(os.environ.get("CONV_TOL", "1e-10"))
mf.max_cycle = 100
nit = [0]
def cb(env): nit[0] += 1
mf.callback = cb
t0 = time.perf_counter()
e = mf.kernel()
if dev == 'gpu':
    cupy.cuda.Device().synchronize()
t1 = time.perf_counter()
print(f"RESULT converged={mf.converged} E={e:.10f} Eh  scf_cycles={nit[0]}  wall={t1-t0:.2f}s")
