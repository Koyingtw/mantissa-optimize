"""Paired 95% t intervals for the QE adaptive-bit repeats (sci_sweep.sbatch jobs merged by sci_summary.py).

usage: qe_adapt_ci.py <sci_summary csv> <fixed variant, e.g. src28:nearest>
Older jobs 1758/1759 ran every variant twice without pass letters, so their pairs are the n-th occurrence of each
variant; newer jobs pair by pass letter.  Each pair is (variant / native - 1) within the same ABBA block.
"""
import sys, pandas as pd, numpy as np
d=pd.read_csv(sys.argv[1]); fixed=sys.argv[2]
d['occ']=d.groupby(['job','variant']).cumcount()
d['pair']=np.where(d.job.str.contains('1758|1759'), d.job+'_'+d.occ.astype(str), d.job+'_'+d['pass'])
T975={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365}
rows=[]
for v in ['adapt0:nearest',fixed]:
    for col in ['wall_s','energy_kj']:
        x=[]
        for p,g in d.groupby('pair'):
            nat=g[g.variant=='native']; var=g[g.variant==v]
            if len(nat) and len(var): x.append(var[col].iloc[0]/nat[col].iloc[0]-1)
        x=np.array(x)*100; n=len(x)
        rows.append(dict(variant=v,measure=col,n=n,mean=round(x.mean(),2),ci95=round(T975[n-1]*x.std(ddof=1)/np.sqrt(n),2),values=np.round(x,2).tolist()))
print(pd.DataFrame(rows).to_string(index=False))
print('能量差（Ry）：', d.groupby('variant').dE.agg(['min','max']).to_string())
print('SCF 圈數：', d.groupby('variant').iters.agg(['min','max']).to_string())
