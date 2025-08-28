import os,pandas as pd,yfinance as yf,joblib,matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.utils import features_from_prices
os.makedirs("data/raw",exist_ok=True)
os.makedirs("data/processed",exist_ok=True)
os.makedirs("model",exist_ok=True)
os.makedirs("reports",exist_ok=True)
tickers=["AAPL","MSFT","SPY"]
df=[]
for t in tickers:
    d=yf.download(t,period="5y",interval="1d",auto_adjust=True,progress=False)
    d["Ticker"]=t
    df.append(d)
df=pd.concat(df).reset_index().rename(columns={"Date":"Date"})
df=df[["Date","Ticker","Open","High","Low","Close","Volume"]]
df.to_csv("data/raw/prices.csv",index=False)
d=df[df["Ticker"]=="SPY"].set_index("Date")
X,y=features_from_prices(d)
X.to_csv("data/processed/features.csv",index=False)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
m=LogisticRegression(max_iter=1000)
m.fit(X_train,y_train)
pred=m.predict(X_test)
open("reports/metrics.txt","w",encoding="utf-8").write(classification_report(y_test,pred))
joblib.dump(m,"model/model.pkl")
plt.figure()
d2=d.copy()
d2["Close"].plot()
plt.tight_layout()
plt.savefig("reports/price_chart.png")
plt.close()
