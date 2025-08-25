import argparse, pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

def feats(d):
    d=d.copy()
    d["lag1"]=d["ret"].shift(1)
    d["lag5"]=d["ret"].shift(5)
    d["roll_mean5"]=d["ret"].shift(1).rolling(5).mean()
    d["roll_std5"]=d["ret"].shift(1).rolling(5).std()
    d["mom5"]=d["price"].pct_change(5).shift(1)
    d["z20"]=((d["ret"].shift(1)-d["ret"].shift(1).rolling(20).mean())/d["ret"].shift(1).rolling(20).std())
    d=d.dropna()
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--out",default="outputs")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.csv,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df=feats(df)
    X=df[["lag1","lag5","roll_mean5","roll_std5","mom5","z20"]].values
    y=df["ret"].shift(-1).dropna()
    X=X[:len(y)]
    n=len(X)
    split=int(n*0.75)
    X_train,X_test=X[:split],X[split:]
    y_train,y_test=y.values[:split],y.values[split:]
    pipe=Pipeline([("scaler",StandardScaler()),("model",Ridge(alpha=1.0))])
    pipe.fit(X_train,y_train)
    p=pipe.predict(X_test)
    mae=mean_absolute_error(y_test,p)
    rmse=mean_squared_error(y_test,p)**0.5
    open(out/"forecast_metrics.txt","w",encoding="utf-8").write(f"MAE={mae:.6f}\nRMSE={rmse:.6f}\n")
    import matplotlib.pyplot as plt
    import numpy as np
    t=np.arange(len(y_test))
    plt.figure(); plt.plot(t,y_test); plt.plot(t,p); plt.title("pred_vs_truth"); plt.savefig(out/"forecast_pred_vs_truth.png",bbox_inches="tight"); plt.close()

if __name__=="__main__":
    main()
