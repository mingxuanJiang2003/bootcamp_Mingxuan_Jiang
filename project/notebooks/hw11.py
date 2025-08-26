import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def find_base():
    c=Path.cwd()
    cand=[c, c.parent, c.parent.parent]
    for b in cand:
        if (b/'project').exists(): return b/'project'
    return c
base=find_base()
out_img=base/'reports'/'images'
out_img.mkdir(parents=True, exist_ok=True)
out_tbl=base/'outputs'
out_tbl.mkdir(parents=True, exist_ok=True)
data1=base/'data'/'processed'/'stage11_eval_data.csv'
data2=base/'data'/'processed'/'stage12_business_metrics.csv'
if data1.exists():
    df=pd.read_csv(data1)
else:
    if data2.exists():
        t=pd.read_csv(data2, parse_dates=['date'])
        g=t.rename(columns={'conversion_rate':'x1','visitors':'x2','revenue':'y'})[['x1','x2','y']].copy()
        g['segment']='A'
        df=g
    else:
        rng=np.random.default_rng(11)
        n=1200
        seg=rng.choice(['A','B','C'], size=n, p=[0.45,0.35,0.20])
        x1=rng.normal(0,1,n)
        x2=rng.normal(0,1.2,n)+np.where(seg=='C',0.6,0)
        miss=rng.random(n)<0.12
        x2[miss]=np.nan
        noise=rng.normal(0, np.where(seg=='C',1.2,0.8), n)
        y=1.5+1.2*x1+0.8*np.nan_to_num(x2, nan=0.0)+np.where(seg=='B',0.5,0)+noise
        df=pd.DataFrame({'segment':seg,'x1':x1,'x2':x2,'y':y})
        (base/'data'/'processed').mkdir(parents=True, exist_ok=True)
        df.to_csv(data1,index=False)
def impute_mean(d,cols):
    g=d.copy()
    for c in cols: g[c]=g[c].fillna(g[c].mean())
    return g
def impute_median(d,cols):
    g=d.copy()
    for c in cols: g[c]=g[c].fillna(g[c].median())
    return g
def drop_missing(d,cols):
    return d.dropna(subset=cols)
def fit_rmse(d,features,target):
    X=d[features].values; y=d[target].values
    m=LinearRegression().fit(X,y)
    p=m.predict(X)
    return float(np.sqrt(mean_squared_error(y,p)))
features=['x1','x2'] if 'x1' in df.columns else ['visitors','conversion_rate']
target='y' if 'y' in df.columns else 'revenue'
if 'segment' not in df.columns: df['segment']='A'
d_mean=impute_mean(df,features)
d_median=impute_median(df,features)
d_drop=drop_missing(df,features)
r_mean=fit_rmse(d_mean,features,target)
r_median=fit_rmse(d_median,features,target)
r_drop=fit_rmse(d_drop,features,target)
plt.figure(); plt.bar(['mean','median','drop'],[r_mean,r_median,r_drop]); plt.title('Stage11 RMSE by Strategy'); plt.ylabel('RMSE'); plt.tight_layout(); plt.savefig(out_img/'stage11_rmse_strategies.png'); plt.close()
seg_res=[]
for s in df['segment'].unique():
    d=d_mean[df['segment']==s]
    seg_res.append((s, fit_rmse(d,features,target)))
seg_tbl=pd.DataFrame(seg_res, columns=['segment','rmse']).sort_values('segment')
seg_tbl.to_csv(out_tbl/'stage11_rmse_by_segment.csv', index=False)
print({'rmse_mean':r_mean,'rmse_median':r_median,'rmse_drop':r_drop})
print('saved:', out_img/'stage11_rmse_strategies.png', out_tbl/'stage11_rmse_by_segment.csv')
