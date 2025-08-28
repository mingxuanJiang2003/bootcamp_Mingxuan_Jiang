from flask import Flask,request,send_file,jsonify
import pandas as pd,numpy as np,io,matplotlib.pyplot as plt,joblib,os,subprocess,sys,datetime
from src.utils import features_from_prices
app=Flask(__name__)
MODEL_PATH=os.path.join("model","model.pkl")
model=None
def load_model():
    global model
    if model is None:
        model=joblib.load(MODEL_PATH)
    return model
@app.route("/predict",methods=["POST"])
def predict_post():
    data=request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error":"features missing"}),400
    X=pd.DataFrame([data["features"]])
    m=load_model()
    p=float(m.predict_proba(X)[0,1])
    y=int(p>=0.5)
    return jsonify({"proba":p,"pred":y})
@app.route("/predict/<x1>")
def predict_get1(x1):
    try:
        v=float(x1)
    except:
        return jsonify({"error":"invalid input"}),400
    X=pd.DataFrame([[v,0,0,0,0,0]],columns=["ret","vol","ma5","ma10","mom5","v_ma5"])
    m=load_model()
    p=float(m.predict_proba(X)[0,1])
    y=int(p>=0.5)
    return jsonify({"proba":p,"pred":y})
@app.route("/predict/<x1>/<x2>")
def predict_get2(x1,x2):
    try:
        a=float(x1);b=float(x2)
    except:
        return jsonify({"error":"invalid input"}),400
    X=pd.DataFrame([[a,b,0,0,0,0]],columns=["ret","vol","ma5","ma10","mom5","v_ma5"])
    m=load_model()
    p=float(m.predict_proba(X)[0,1])
    y=int(p>=0.5)
    return jsonify({"proba":p,"pred":y})
@app.route("/plot")
def plot():
    import numpy as np
    img=io.BytesIO()
    t=pd.date_range(end=datetime.date.today(),periods=60)
    s=pd.Series(np.random.randn(60).cumsum(),index=t)
    ax=s.plot()
    plt.tight_layout()
    plt.savefig(img,format="png")
    plt.close()
    img.seek(0)
    return send_file(img,mimetype="image/png")
@app.route("/run_full_analysis")
def run_full_analysis():
    try:
        r=subprocess.run([sys.executable,"notebooks/hw13.py"],capture_output=True,text=True,timeout=600)
        if r.returncode!=0:
            return jsonify({"error":"pipeline failed","stderr":r.stderr}),500
        global model
        model=None
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"error":"exception"}),500
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
