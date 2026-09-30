import csv,sys,re
rows=list(csv.DictReader(open(sys.argv[1])))
tot=sum(float(r['Total Time (ns)']) for r in rows)
def isgemm(n):
    n=n.lower()
    return any(k in n for k in ('gemm','cutlass','xmma','sm90_','nvjet','gemv','trsm','syrk','herk'))
g=[r for r in rows if isgemm(r['Name'])]
gt=sum(float(r['Total Time (ns)']) for r in g)
wall=None
if len(sys.argv)>2:
    try:
        t=open(sys.argv[2]).read()
        m=re.findall(r'PWSCF\s*:.*?CPU\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)s WALL',t)
        if m:
            h,mi,se=m[-1]; wall=float(h or 0)*3600+float(mi or 0)*60+float(se)
    except Exception: pass
print(f"GPU kernel total = {tot/1e9:.2f} s ; BLAS-like kernels (gemm/gemv/trsm/...) = {gt/1e9:.2f} s = {100*gt/tot:.1f}% of GPU kernel time")
if wall: print(f"wall = {wall:.2f} s ; BLAS kernels / wall = {100*gt/1e9/wall:.1f}%")
gg=sum(float(r['Total Time (ns)']) for r in rows if 'gemm' in r['Name'].lower() or 'cutlass' in r['Name'].lower() or 'nvjet' in r['Name'].lower())
print(f"GEMM-only kernels = {gg/1e9:.2f} s = {100*gg/tot:.1f}% of GPU kernel time" + (f" = {100*gg/1e9/wall:.1f}% of wall" if wall else ""))
print("Top 12 kernels:")
for r in sorted(rows,key=lambda r:-float(r['Total Time (ns)']))[:12]:
    print(f"  {float(r['Total Time (ns)'])/1e9:8.3f}s {float(r['Total Time (ns)'])/tot*100:5.1f}% n={r['Instances']:>7} {'[BLAS]' if isgemm(r['Name']) else '      '} {r['Name'][:110]}")
