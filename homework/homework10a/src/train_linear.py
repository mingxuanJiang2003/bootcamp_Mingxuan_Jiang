import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--features", nargs="*", default=None, help="Optional explicit feature list")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.features is None or len(args.features) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != args.target]
    else:
        features = args.features

    X = df[features].copy()
    y = df[args.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test,  y_test_pred)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test  = mean_squared_error(y_test,  y_test_pred,  squared=False)

    print("Features:", features)
    print("Coefficients:", dict(zip(features, model.coef_)))
    print("Intercept:", model.intercept_)
    print(f"R2_train={r2_train:.4f}  R2_test={r2_test:.4f}")
    print(f"RMSE_train={rmse_train:.4f}  RMSE_test={rmse_test:.4f}")

    (out_dir / "metrics.txt").write_text(
        f"R2_train={r2_train:.6f}\nR2_test={r2_test:.6f}\nRMSE_train={rmse_train:.6f}\nRMSE_test={rmse_test:.6f}\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
