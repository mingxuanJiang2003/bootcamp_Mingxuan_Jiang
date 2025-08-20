import pandas as pd
def to_datetime(df: pd.DataFrame, col: str, fmt: str|None=None) -> pd.DataFrame:
    df=df.copy(); df[col]=pd.to_datetime(df[col], format=fmt, errors="coerce"); return df
