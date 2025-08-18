# fetch_market_data.py
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import datetime
from pathlib import Path

# Date range (5 years)
end = datetime.datetime.today()
start = end - datetime.timedelta(days=5*365)

# 1) SPY data from Yahoo Finance
spy = yf.download("SPY", start=start, end=end)
spy = spy.reset_index()
spy = spy.rename(columns={
    "Date": "date",
    "Adj Close": "spy_close",
    "Open": "spy_open",
    "High": "spy_high",
    "Low": "spy_low",
    "Close": "spy_close_raw",
    "Volume": "spy_volume"
})
spy = spy[["date","spy_open","spy_high","spy_low","spy_close","spy_volume"]]

# 2) VIX data from FRED
vix = pdr.DataReader("VIXCLS", "fred", start, end).reset_index()
vix = vix.rename(columns={"DATE":"date","VIXCLS":"vix_close"})

# 3) Merge
df = pd.merge(spy, vix, on="date", how="inner")
df["spy_return"] = df["spy_close"].pct_change()
df["rv_5d"] = df["spy_return"].rolling(5).std() * (252**0.5)
df["rv_21d"] = df["spy_return"].rolling(21).std() * (252**0.5)

# 4) Save
out_path = Path("data/market_daily_real.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved real-world dataset to {out_path.resolve()}")
