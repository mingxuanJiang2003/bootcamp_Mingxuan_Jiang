import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pathlib import Path
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True)
    ap.add_argument("--target",required=True)
    ap.add_argument("--out",default="project/outputs")
    args=ap.parse_args()
    p=Path(args.out); p.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.csv)
    num=df.select_dtypes(include=[np.number]).columns.tolist()
    feats=[c for c in num if c!=args.target]
    X=df[feats].values
    y=df[args.target].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
    model=LinearRegression().fit(Xtr,ytr)
    ytep=model.predict(Xte)
    resid=yte-ytep
    plt.figure(); plt.scatter(ytep,resid); plt.axhline(0,linestyle="--"); plt.title("residuals_vs_fitted"); plt.savefig(p/"hw10a_resid_vs_fitted.png",bbox_inches="tight"); plt.close()
    plt.figure(); plt.hist(resid,bins=30); plt.title("residual_hist"); plt.savefig(p/"hw10a_resid_hist.png",bbox_inches="tight"); plt.close()
    import scipy.stats as st
    plt.figure(); st.probplot(resid, dist="norm", plot=plt); plt.title("qq_resid"); plt.savefig(p/"hw10a_resid_qq.png",bbox_inches="tight"); plt.close()
if __name__=="__main__":
    main()
