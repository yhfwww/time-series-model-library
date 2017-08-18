import pandas as pd
import numpy as np
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
#user_balance= pd.read_csv(r'user_balance_table.csv')
dateparse=lambda dates:pd.datetime.strptime(dates,'%Y%m%d')
user_balance= pd.read_csv(r'user_balance_table.csv',parse_dates=True,index_col='report_date',date_parser=dateparse)
user_balance=user_balance.fillna(0)
#df = user_balance.groupby(by=['report_date'])['column_B'].sum()
df = user_balance.groupby(by=['report_date']).sum()
#total_purchase_amt=df['total_purchase_amt']
#total_redeem_amt=df['total_redeem_amt']
#total_purchase_amt+++
#total_redeem_amt---
test=df['total_redeem_amt'][-31:]
df=df#[0:-31]

model = pf.ARIMA(data=df,ar=8,ma=5,target='total_redeem_amt',family=pf.Normal())
x=model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))
model.plot_predict_is(h=80, figsize=(15,5))
#画预测图
model.plot_predict(h=31,past_values=9,figsize=(15,5))
pre=model.predict(31)#.values.reshape(31,)

testval=test.values
dif=pre-testval
pd.DataFrame([dif,pre,testval]).T.plot()
err=abs(dif).sum()
print(err)
