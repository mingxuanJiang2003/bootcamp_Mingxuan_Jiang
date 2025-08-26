import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def impute_mean(df,cols):
    g=df.copy()
    for c in cols:
        g[c]=g[c].fillna(g[c].mean())
    return g
def impute_median(df,cols):
    g=df.copy()
    for c in cols:
        g[c]=g[c].fillna(g[c].median())
    return g
def drop_missing(df,cols):
    return df.dropna(subset=cols)
def fit_and_rmse(df,features,target):
    X=df[features].values
    y=df[target].values
    m=LinearRegression()
    m.fit(X,y)
    p=m.predict(X)
    return float(np.sqrt(mean_squared_error(y,p))), m
def bootstrap_rmse(df,features,target,n_boot,random_state=0):
    rng=np.random.default_rng(random_state)
    n=len(df)
    out=np.empty(n_boot)
    for b in range(n_boot):
        idx=rng.integers(0,n,size=n)
        s=df.iloc[idx]
        out[b]=fit_and_rmse(s,features,target)[0]
    return out
def rmse_by_segment(df,features,target,segment_col):
    res=[]
    for seg in df[segment_col].unique():
        d=df[df[segment_col]==seg]
        r,_=fit_and_rmse(d,features,target)
        res.append((seg,r))
    return pd.DataFrame(res,columns=[segment_col,"rmse"]).sort_values(segment_col)
