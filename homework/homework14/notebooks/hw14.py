import os
os.makedirs("handoff", exist_ok=True)
from PIL import Image,ImageDraw
def box(d,x,y,w,h,t):
    d.rectangle([x,y,x+w,y+h],outline="black",width=3)
    d.text((x+10,y+10),t,fill="black")
img=Image.new("RGB",(1200,800),"white")
d=ImageDraw.Draw(img)
box(d,30,30,1140,80,"Monitoring Dashboard Wireframe")
box(d,30,130,560,260,"Data: Freshness, Nulls, Schema Hash, PSI")
box(d,610,130,560,260,"Model: Rolling AUC, Brier, Calibration")
box(d,30,410,560,260,"System: p95 Latency, Error Rate, Uptime")
box(d,610,410,560,260,"Business: Usage, Approval, PnL Contribution")
img.save("handoff/dashboard_sketch.png")
print("handoff/dashboard_sketch.png")
