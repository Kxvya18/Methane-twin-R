import numpy as np

def bootstrap_ci(values, statistic=np.mean, n_boot=2000, alpha=0.05, seed=42):
    x=np.asarray(values,float); rng=np.random.default_rng(seed)
    if len(x)==0:return (None,None)
    stats=[statistic(rng.choice(x,size=len(x),replace=True)) for _ in range(n_boot)]
    return float(np.quantile(stats,alpha/2)),float(np.quantile(stats,1-alpha/2))
