import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
def find_base():
    c=Path.cwd()
    cand=[c, c.parent, c.parent.parent]
    for b in cand:
        if (b/'project').exists(): return b/'project'
    return c
base=find_base()
out_img=base/'reports'/'images'
out_img.mkdir(parents=True, exist_ok=True)
out_tbl=base/'outputs'
out_tbl.mkdir(parents=True, exist_ok=True)
data=base/'data'/'processed'/'stage12_business_metrics.csv'
if not data.exists():
    import numpy as np, pandas as pd
    rng=np.random.default_rng(12)
    days=220
    dates=pd.date_range('2025-01-01', periods=days, freq='D')
    seg=rng.choice(['A','B','C'], size=days, p=[0.5,0.35,0.15])
    visitors=(rng.normal(1200,200,days)).clip(400).astype(int)
    base_conv=np.where(seg=='A',0.042,np.where(seg=='B',0.037,0.030))
    season=0.004*np.sin(np.linspace(0,8*np.pi,days))
    conv=(base_conv+season+rng.normal(0,0.003,days)).clip(0.015,0.06)
    conv_n=np.round(visitors*conv).astype(int)
    price=np.where(seg=='C',45,np.where(seg=='B',48,50))
    discount=np.where((pd.Series(dates).dt.month%3==0),0.95,1.0)
    revenue=(conv_n*price*discount*(1+rng.normal(0,0.02,days))).clip(0)
    cac=np.where(seg=='A',6.0,np.where(seg=='B',6.5,7.5))
    ad_spend=(visitors*0.25 + conv_n*cac + rng.normal(0,60,days)).clip(100)
    unit_cost=np.where(seg=='C',24,np.where(seg=='B',22,20))
    cogs=conv_n*unit_cost*(1+rng.normal(0,0.015,days))
    profit=revenue - ad_spend - cogs
    df=pd.DataFrame({'date':dates,'segment':seg,'visitors':visitors,'conversions':conv_n,'conversion_rate':conv,'price':price,'revenue':revenue,'ad_spend':ad_spend,'cogs':cogs,'profit':profit})
    (base/'data'/'processed').mkdir(parents=True, exist_ok=True)
    df.to_csv(data,index=False)
else:
    df=pd.read_csv(data, parse_dates=['date'])
rev=df.set_index('date')['revenue'].rolling(7).mean()
plt.figure(); plt.plot(rev.index, rev.values); plt.title('Revenue (7D Rolling)'); plt.xlabel('Date'); plt.ylabel('Revenue'); plt.tight_layout(); plt.savefig(out_img/'stage12_line_revenue_trend.png'); plt.close()
conv_by_seg=df.groupby('segment')['conversion_rate'].mean().reindex(['A','B','C'])
plt.figure(); plt.bar(conv_by_seg.index, conv_by_seg.values); plt.title('Avg Conversion Rate by Segment'); plt.xlabel('Segment'); plt.ylabel('Conversion Rate'); plt.tight_layout(); plt.savefig(out_img/'stage12_bar_conversion_by_segment.png'); plt.close()
wk=df.resample('W-MON', on='date').agg({'ad_spend':'sum','revenue':'sum'}).reset_index()
plt.figure(); plt.scatter(wk['ad_spend'], wk['revenue']); plt.title('Weekly Revenue vs Ad Spend'); plt.xlabel('Ad Spend'); plt.ylabel('Revenue'); plt.tight_layout(); plt.savefig(out_img/'stage12_scatter_spend_vs_revenue.png'); plt.close()
monthly=df.resample('MS', on='date').agg({'conversions':'sum','price':'mean','ad_spend':'sum','cogs':'sum','revenue':'sum'})
baseline=(monthly['revenue']-monthly['ad_spend']-monthly['cogs']).mean()
def scen(d,cm=1.0,pm=1.0,um=1.0,sm=1.0):
    m=d.copy()
    conv=m['conversions']*cm
    price=m['price']*pm
    rev=conv*price
    cogs=(conv*(m['cogs']/m['conversions'].replace(0,np.nan)).fillna(0))*um
    spend=m['ad_spend']*sm
    return (rev-spend-cogs).mean()
names=['Baseline','Conv -5%','Price +5%','UnitCost +10%','AdSpend -10%']
pars=[(1.0,1.0,1.0,1.0),(0.95,1.0,1.0,1.0),(1.0,1.05,1.0,1.0),(1.0,1.0,1.10,1.0),(1.0,1.0,1.0,0.90)]
vals=[scen(monthly,*t) for t in pars]
res=pd.DataFrame({'scenario':names,'avg_monthly_profit':vals})
res['delta_vs_baseline']=res['avg_monthly_profit']-baseline
res.to_csv(out_tbl/'stage12_sensitivity_table.csv', index=False)
plt.figure(); plt.bar(res['scenario'], res['delta_vs_baseline']); plt.title('Sensitivity: Δ Profit vs Baseline'); plt.xlabel('Scenario'); plt.ylabel('Δ Profit'); plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.savefig(out_img/'stage12_bar_sensitivity_delta_profit.png'); plt.close()
print('saved:', out_img, out_tbl/'stage12_sensitivity_table.csv')
