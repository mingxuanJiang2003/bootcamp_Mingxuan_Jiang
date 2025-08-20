import pandas as pd
def detect_outliers_iqr(s: pd.Series, k: float=1.5) -> pd.Series:
    q1=s.quantile(0.25); q3=s.quantile(0.75); iqr=q3-q1; lo=q1-k*iqr; hi=q3+k*iqr
    return (s<lo)|(s>hi)
def detect_outliers_zscore(s: pd.Series, threshold: float=3.0) -> pd.Series:
    mu=s.mean(); sd=s.std() if s.std()!=0 else 1.0; z=(s-mu)/sd; return z.abs()>threshold
def winsorize(s: pd.Series, lower: float=0.05, upper: float=0.95) -> pd.Series:
    lo=s.quantile(lower); hi=s.quantile(upper); return s.clip(lower=lo, upper=hi)
