import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--features", nargs="*", default=None, help="Optional explicit feature list")
    ap.add_argument("--out", default="outputs", help="Output directory for plots")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if args.target not in numeric_cols:
        raise ValueError(f"Target '{args.target}' must be numeric. Numeric cols: {numeric_cols}")

    if args.features is None or len(args.features) == 0:
        features = [c for c in numeric_cols if c != args.target]
    else:
        features = args.features

    X = df[features].copy()
    y = df[args.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    resid = y_test - y_test_pred

    plt.figure()
    plt.scatter(y_test_pred, resid, alpha=0.8)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values (test)")
    plt.ylabel("Residuals (test)")
    plt.title("Residuals vs Fitted")
    plt.savefig(out_dir / "residuals_vs_fitted.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(resid, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Histogram (test)")
    plt.savefig(out_dir / "residual_hist.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("QQ Plot of Residuals (test)")
    plt.savefig(out_dir / "residual_qq.png", bbox_inches="tight")
    plt.close()

    key_pred = X_test.columns[0]
    plt.figure()
    plt.scatter(X_test[key_pred], resid, alpha=0.8)
    plt.axhline(0, linestyle="--")
    plt.xlabel(key_pred + " (test)")
    plt.ylabel("Residuals (test)")
    plt.title(f"Residuals vs {key_pred}")
    plt.savefig(out_dir / f"residuals_vs_{key_pred}.png", bbox_inches="tight")
    plt.close()

    r = resid.reset_index(drop=True)
    r_lag = r.shift(1)
    valid = r_lag.notna()
    corr = np.corrcoef(r[valid], r_lag[valid])[0,1]

    plt.figure()
    plt.scatter(r[valid], r_lag[valid], alpha=0.8)
    plt.xlabel("Residual[t]")
    plt.ylabel("Residual[t-1]")
    plt.title(f"Lag-1 Residual Scatter (corr ≈ {corr:.3f})")
    plt.savefig(out_dir / "residual_lag1.png", bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
