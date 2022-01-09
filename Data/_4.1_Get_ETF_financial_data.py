import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time


apiKey = 'TCBN46GY5MD7ASKD'
ts = TimeSeries(key = apiKey, output_format = 'csv')
app = TimeSeries(key = apiKey, output_format = 'pandas')




#Get CARZ data
#----------------
finance_data_short_CARZ, finance_data_short_CARZ_meta_data = app.get_daily_adjusted(symbol = 'CARZ', outputsize = 'full')
finance_data_short_CARZ['date']=finance_data_short_CARZ.index
finance_data_short_CARZ.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_short_CARZ.csv', index = False)

data= pd.DataFrame()
i=1

while i <=12:
    slice='year1'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'CARZ', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    if i != 1:
        df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year2'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'CARZ', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year3'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'CARZ', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1



data.dropna(axis=1, how="all", thresh=None)
header = data.iloc[0]
finance_data_extended_CARZ = data[1:]
finance_data_extended_CARZ.columns =header
finance_data_extended_CARZ.set_index('time')



finance_data_extended_CARZ.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_CARZ.csv', index = False)

"""
#Get GM data
#-----------------------
finance_data_short_GM, finance_data_short_GM_meta_data = app.get_daily_adjusted(symbol = 'GM', outputsize = 'full')
finance_data_short_GM['date']=finance_data_short_GM.index
finance_data_short_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data//finance_data_short_GM.csv', index = False)

data= pd.DataFrame()
i=1

while i <=12:
    slice='year1'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'GM', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    if i != 1:
        df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year2'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'GM', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1



data.dropna(axis=1, how="all", thresh=None)
header = data.iloc[0]
finance_data_extended_GM = data[1:]
finance_data_extended_GM.columns =header
finance_data_extended_GM.set_index('time')

finance_data_extended_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/3. Finance Data/finance_data_extended_GM.csv', index = False)
"""
