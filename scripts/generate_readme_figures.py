from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'docs'/'assets'/'data'; FIG=ROOT/'docs'/'assets'/'figures'
FIG.mkdir(exist_ok=True)
S=pd.read_csv(DATA/'summary_with_adaptive.csv')
R=pd.read_csv(DATA/'all_raw_with_adaptive.csv')
G=pd.read_csv(DATA/'burst_scores.csv')
E=pd.read_csv(DATA/'exact_small_summary.csv')
A=pd.read_csv(DATA/'pool_ablation_summary.csv')
D=pd.read_csv(DATA/'distributed_map_summary.csv')
methods=['Adaptive-SPARSE','MP','FISTA','IRLS','OMP','OLS','FIFO-Windowed']
labels={'Adaptive-SPARSE':'Adaptive','FIFO-Windowed':'FCFS','LangChain-FCFS-policy':'LangChain policy','LangGraph-FCFS-policy':'LangGraph policy'}
def save(name):
 plt.tight_layout(); plt.savefig(FIG/f'{name}.pdf',bbox_inches='tight'); plt.savefig(FIG/f'{name}.png',dpi=220,bbox_inches='tight'); plt.close()
# 1m makespan
x=S[(S.pattern=='correlated_bursts')&(S.n_agents==1000000)&S.method.isin(methods)].copy(); x['ord']=x.method.map({m:i for i,m in enumerate(methods)}); x=x.sort_values('ord')
plt.figure(figsize=(8.6,4.5)); plt.bar(range(len(x)),x.norm_mean,yerr=x.norm_std,capsize=4); plt.axhline(1,ls='--',lw=1); plt.xticks(range(len(x)),[labels.get(m,m) for m in x.method],rotation=20,ha='right'); plt.ylabel('normalized makespan'); plt.title('One million explicit agents: correlated-burst stress'); save('result_1m_makespan')
# 1m time
plt.figure(figsize=(8.6,4.5)); plt.bar(range(len(x)),x.time_mean,yerr=x.time_std,capsize=4); plt.xticks(range(len(x)),[labels.get(m,m) for m in x.method],rotation=20,ha='right'); plt.ylabel('scheduler time (s)'); plt.yscale('log'); plt.title('Control-plane cost at one million agents'); save('result_1m_time')
# scale quality
plt.figure(figsize=(8.3,4.8))
for m in ['Adaptive-SPARSE','MP','FISTA','IRLS','OMP','FIFO-Windowed']:
 q=S[(S.pattern=='correlated_bursts')&S.method.eq(m)&S.n_agents.isin([10000,100000,1000000])].sort_values('n_agents')
 plt.errorbar(q.n_agents,q.norm_mean,yerr=q.norm_std,marker='o',label=labels.get(m,m),capsize=3)
plt.xscale('log'); plt.xlabel('number of explicit agents'); plt.ylabel('normalized makespan'); plt.legend(ncol=3,fontsize=8); plt.title('Quality scaling under correlated arrivals'); save('scale_quality')
# scale time
plt.figure(figsize=(8.3,4.8))
for m in ['Adaptive-SPARSE','MP','FISTA','IRLS','OMP','OLS']:
 q=S[(S.pattern=='correlated_bursts')&S.method.eq(m)&S.n_agents.isin([10000,100000,1000000])].sort_values('n_agents')
 plt.errorbar(q.n_agents,q.time_mean,yerr=q.time_std,marker='o',label=labels.get(m,m),capsize=3)
plt.xscale('log'); plt.yscale('log'); plt.xlabel('number of explicit agents'); plt.ylabel('scheduler time (s)'); plt.legend(ncol=3,fontsize=8); plt.title('Control-plane scaling'); save('scale_time')
# regimes
reg=[]
for pat,n,title in [('correlated_bursts',100000,'bursts'),('iid',100000,'IID'),('complementary',100000,'complementary')]:
 for m in ['Adaptive-SPARSE','MP','FISTA','FIFO-Windowed']:
  z=S[(S.pattern==pat)&(S.n_agents==n)&(S.method==m)].iloc[0]
  reg.append((title,m,z.norm_mean,z.norm_std))
plt.figure(figsize=(8.5,4.7)); width=.19; cats=['bursts','IID','complementary']; xx=np.arange(3)
for j,m in enumerate(['Adaptive-SPARSE','MP','FISTA','FIFO-Windowed']):
 vals=[next(v for c,mm,v,s in reg if c==c0 and mm==m) for c0 in cats]; errs=[next(s for c,mm,v,s in reg if c==c0 and mm==m) for c0 in cats]
 plt.bar(xx+(j-1.5)*width,vals,width,yerr=errs,capsize=2,label=labels.get(m,m))
plt.xticks(xx,cats); plt.ylabel('normalized makespan'); plt.legend(ncol=4,fontsize=8); plt.title('Positive stress and negative controls'); save('regime_controls')
# gate scores
plt.figure(figsize=(8.2,4.5)); groups=[]
for pat,n,title in [('correlated_bursts',1000000,'bursts, 1M'),('iid',100000,'IID, 100k'),('complementary',100000,'complementary, 100k')]:
 vals=G[(G.pattern==pat)&(G.n_agents==n)].burst_score.values; groups.append(vals)
plt.boxplot(groups,labels=['bursts, 1M','IID, 100k','complementary, 100k'],showmeans=True); plt.axhline(.1,ls='--',label='gate threshold'); plt.ylabel('order-correlation excess'); plt.legend(); plt.title('Adaptive gate separates queue regimes'); save('burst_gate')
# per seed 1m
z=R[(R.pattern=='correlated_bursts')&(R.n_agents==1000000)&R.method.isin(['MP','FISTA','FIFO-Windowed'])]
plt.figure(figsize=(8.5,4.5))
for m in ['MP','FISTA','FIFO-Windowed']:
 q=z[z.method==m].sort_values('seed'); plt.plot(q.seed,q.normalized_makespan,marker='o',label=labels.get(m,m))
plt.xlabel('held-out seed'); plt.ylabel('normalized makespan'); plt.legend(); plt.title('Per-instance stability at one million agents'); save('per_seed_1m')
# exact recovery
q=E[E.method.isin(['MP','OMP','OLS','FISTA','IRLS','FIFO-Windowed'])]
piv=q.pivot(index='method',columns='pattern',values='exact_rate').reindex(['MP','OMP','OLS','FISTA','IRLS','FIFO-Windowed'])
plt.figure(figsize=(8.4,4.6)); width=.25; xx=np.arange(len(piv))
for j,col in enumerate(['correlated_bursts','iid','complementary']): plt.bar(xx+(j-1)*width,piv[col].values,width,label=col.replace('_',' '))
plt.xticks(xx,[labels.get(m,m) for m in piv.index]); plt.ylabel('exact optimum recovery rate'); plt.ylim(0,1); plt.legend(fontsize=8); plt.title('MILP-certified small instances'); save('exact_recovery')
# pool ablation
plt.figure(figsize=(8.2,4.5)); ax=plt.gca(); ax.errorbar(A.pool_size,A.norm_mean,yerr=A.norm_std,marker='o',capsize=3); ax.set_xscale('log'); ax.set_xlabel('candidate pool size'); ax.set_ylabel('normalized makespan'); ax2=ax.twinx(); ax2.plot(A.pool_size,A.time_mean,marker='s',ls='--'); ax2.set_ylabel('scheduler time (s)'); plt.title('Candidate-pool ablation at 100k agents'); save('pool_ablation')
# distributed map
plt.figure(figsize=(8.2,4.5)); plt.errorbar(D.workers,D.wall_mean,yerr=D.wall_std,marker='o',capsize=3); plt.xlabel('worker processes'); plt.ylabel('top-k map-reduce wall time (s)'); plt.xticks(D.workers); plt.title('Single-node process backend: overhead dominates'); save('distributed_map')
# quality cost 1m
q=S[(S.pattern=='correlated_bursts')&(S.n_agents==1000000)&S.method.isin(['MP','OMP','OLS','FISTA','IRLS','FIFO-Windowed'])]
plt.figure(figsize=(7.5,5.2)); plt.scatter(q.time_mean,q.norm_mean,s=70)
for _,r in q.iterrows(): plt.annotate(labels.get(r.method,r.method),(r.time_mean,r.norm_mean),xytext=(5,4),textcoords='offset points',fontsize=8)
plt.xscale('log'); plt.xlabel('scheduler time (s, log scale)'); plt.ylabel('normalized makespan'); plt.title('Quality-cost frontier at one million agents'); save('quality_cost_1m')
# improvement scale
q=[]
for n in [10000,100000,1000000]:
 a=S[(S.pattern=='correlated_bursts')&(S.n_agents==n)&(S.method=='MP')].iloc[0].norm_mean
 b=S[(S.pattern=='correlated_bursts')&(S.n_agents==n)&(S.method=='FIFO-Windowed')].iloc[0].norm_mean
 q.append(100*(1-a/b))
plt.figure(figsize=(7.8,4.3)); plt.plot([10000,100000,1000000],q,marker='o'); plt.xscale('log'); plt.xlabel('number of explicit agents'); plt.ylabel('mean makespan reduction (%)'); plt.ylim(0,60); plt.title('MP improvement over FCFS remains near 50%'); save('improvement_scale')
