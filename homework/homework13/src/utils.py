import pandas as pd
import numpy as np
def make_features(df):
    df=df.copy()
    df["ret"]=df["Close"].pct_change()
    df["vol"]=df["Close"].rolling(5).std()
    df["ma5"]=df["Close"].rolling(5).mean()
    df["ma10"]=df["Close"].rolling(10).mean()
    df["mom5"]=df["Close"].pct_change(5)
    df["v_ma5"]=df["Volume"].rolling(5).mean()
    df["target"]=(df["Close"].shift(-1)>df["Close"]).astype(int)
    df=df.dropna()
    X=df[["ret","vol","ma5","ma10","mom5","v_ma5"]]
    y=df["target"]
    return X,y
