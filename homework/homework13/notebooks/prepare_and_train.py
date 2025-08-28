import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib,os
from src.utils import make_features
tickers=["AAPL","MSFT","SPY"]
df=[]
for t in tickers:
    d=yf.download(t,period="5y",interval="1d",auto_adjust=True,progress=False)
    d["Ticker"]=t
    df.append(d)
df=pd.concat(df).reset_index().rename(columns={"Date":"Date"})
df=df[["Date","Ticker","Open","High","Low","Close","Volume"]]
df.to_csv(os.path.join("data","prices.csv"),index=False)
d=df[df["Ticker"]=="SPY"].set_index("Date")
X,y=make_features(d)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
m=LogisticRegression(max_iter=1000)
m.fit(X_train,y_train)
pred=m.predict(X_test)
print(classification_report(y_test,pred))
os.makedirs("model",exist_ok=True)
joblib.dump(m,os.path.join("model","model.pkl"))
