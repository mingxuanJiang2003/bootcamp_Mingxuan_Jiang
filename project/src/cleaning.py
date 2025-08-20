import pandas as pd, numpy as np
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy(); df.columns=[c.strip().lower().replace(" ","_") for c in df.columns]; return df
def fill_na_num(df: pd.DataFrame, cols: list[str], value: float=0.0) -> pd.DataFrame:
    df=df.copy(); 
    for c in cols: df[c]=pd.to_numeric(df[c], errors="coerce").fillna(value)
    return df
def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df=df.copy()
    for c in cols:
        x=pd.to_numeric(df[c], errors="coerce")
        mu=x.mean(); sd=x.std() if x.std()!=0 else 1.0
        df[c]=(x-mu)/sd
    return df
