import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
def feats(d):
    d=d.copy()
    d["lag1"]=d["ret"].shift(1)
    d["lag5"]=d["ret"].shift(5)
    d["roll_mean5"]=d["ret"].shift(1).rolling(5).mean()
    d["roll_std5"]=d["ret"].shift(1).rolling(5).std()
    d["mom5"]=d["price"].pct_change(5).shift(1)
    d["z20"]=(d["ret"].shift(1)-d["ret"].shift(1).rolling(20).mean())/d["ret"].shift(1).rolling(20).std()
    d=d.dropna()
    return d
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--out",default="project/outputs")
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df=feats(df)
    X=df[["lag1","lag5","roll_mean5","roll_std5","mom5","z20"]].values
    y=(df["ret"].shift(-1)>0).astype(int).dropna().values
    X=X[:len(y)]
    n=len(X)
    split=int(n*0.75)
    Xtr,Xte=X[:split],X[split:]
    ytr,yte=y[:split],y[split:]
    pipe=Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=200))])
    pipe.fit(Xtr,ytr)
    p=pipe.predict(Xte)
    acc=accuracy_score(yte,p)
    prec=precision_score(yte,p,zero_division=0)
    rec=recall_score(yte,p,zero_division=0)
    f1=f1_score(yte,p,zero_division=0)
    open(out/"hw10b_classify_metrics.txt","w",encoding="utf-8").write(f"ACC={acc:.6f}\nPREC={prec:.6f}\nREC={rec:.6f}\nF1={f1:.6f}\n")
    cm=confusion_matrix(yte,p); plt.figure(); plt.imshow(cm); plt.title("confusion_matrix"); plt.savefig(out/"hw10b_confusion_matrix.png",bbox_inches="tight"); plt.close()
if __name__=="__main__":
    main()
