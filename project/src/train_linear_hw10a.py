import argparse, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--target",required=True)
    ap.add_argument("--out",default="project/outputs")
    args=ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.csv)
    num=df.select_dtypes(include=[np.number]).columns.tolist()
    feats=[c for c in num if c!=args.target]
    X=df[feats].values
    y=df[args.target].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    model=LinearRegression().fit(Xtr,ytr)
    ytrp=model.predict(Xtr)
    ytep=model.predict(Xte)
    r2tr=r2_score(ytr,ytrp)
    r2te=r2_score(yte,ytep)
    rmsetr=mean_squared_error(ytr,ytrp, squared=True)**0.5
    rmsete=mean_squared_error(yte,ytep, squared=True)**0.5
    open(Path(args.out)/"hw10a_metrics.txt","w",encoding="utf-8").write(f"R2_train={r2tr:.6f}\nR2_test={r2te:.6f}\nRMSE_train={rmsetr:.6f}\nRMSE_test={rmsete:.6f}\n")
if __name__=="__main__":
    main()
