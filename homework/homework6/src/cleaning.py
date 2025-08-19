from __future__ import annotations
from typing import Iterable, Optional, List
import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    MinMaxScaler = None

def _to_list(columns: Optional[Iterable]) -> Optional[List[str]]:
    if columns is None:
        return None
    if isinstance(columns, (list, tuple, set, pd.Index)):
        return list(columns)
    return [columns]

def fill_missing_median(df: pd.DataFrame, columns: Optional[Iterable] = None) -> pd.DataFrame:
    result = df.copy()
    cols = (result.select_dtypes(include=["number"]).columns.tolist()
            if columns is None else _to_list(columns))
    for col in cols:
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
            median = result[col].median(skipna=True)
            result[col] = result[col].fillna(median)
    return result

def drop_missing(df: pd.DataFrame, threshold: float = 0.5, axis: str = "columns") -> pd.DataFrame:
    if axis == "columns":
        miss = df.isna().mean(axis=0)
        to_drop = miss[miss > threshold].index
        return df.drop(columns=to_drop)
    else:
        miss = df.isna().mean(axis=1)
        keep = miss <= threshold
        return df.loc[keep].copy()

def normalize_data(df: pd.DataFrame, columns: Optional[Iterable] = None) -> pd.DataFrame:
    result = df.copy()
    cols = (result.select_dtypes(include=["number"]).columns.tolist()
            if columns is None else _to_list(columns))
    cols = [c for c in cols if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
    if not cols:
        return result
    if MinMaxScaler is not None:
        scaler = MinMaxScaler()
        result[cols] = scaler.fit_transform(result[cols])
        return result
    for col in cols:
        col_min = result[col].min(skipna=True)
        col_max = result[col].max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max) or (col_max - col_min) == 0:
            result[col] = 0.0
        else:
            result[col] = (result[col] - col_min) / (col_max - col_min)
    return result

