

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

##################################
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime,timedelta
import pandas as pd
##################################

_stock_AV = json.loads(open('data\\stock_data_AV.json').read())
_stock_YF = json.loads(open('data\\stock_data_YF.json').read())

###############################################################################
'''problematic data'''
del _stock_AV['LUMN']
del _stock_AV['DNR']
del _stock_AV['LM']
del _stock_AV['NE']

del _stock_YF['LUMN']
del _stock_YF['DNR']
del _stock_YF['LM']
del _stock_YF['NE']

###############################################################################
dat = {} 
for ticker in tqdm(_stock_AV.keys()):
    AV = pd.DataFrame.from_dict(_stock_AV[ticker]).T
    AV = AV.loc[ pd.to_datetime(AV.index) <= datetime(2020, 9, 30, 0, 0)]
    AV = AV.loc[ pd.to_datetime(AV.index) >= datetime(1999, 11, 1, 0, 0)]
    
    YF = pd.DataFrame.from_dict(_stock_YF[ticker]).T
    YF = YF.loc[ pd.to_datetime(YF.index) <= datetime(2020, 9, 30, 0, 0)]
    YF = YF.loc[ pd.to_datetime(YF.index) >= datetime(1999, 11, 1, 0, 0)]
    
    merge = pd.merge(YF,AV,left_index=True,right_index=True,how = "left")
    merge = merge.astype(float)
    
    merge['Date'] = pd.to_datetime(merge.index)
    merge = merge.sort_values(by = ['Date'])
    
    merge = merge[np.logical_and(merge['5. adjusted close']*1.02 >= merge['Adj Close'],
                                 merge['5. adjusted close']*0.98 <= merge['Adj Close'])]    
    merge = merge[merge['Adj Close'] != 0]
    
    '''10 days of gaping allowed'''
    temp = np.where( merge['Date'] - merge.shift()['Date'] < timedelta(days=-10))[0]
    if len(temp) > 0:
        merge = merge.iloc[0:temp[0]]  
     
    merge = merge[['Adj Close','Volume','Open', 'High', 'Low', 'Close']]    
    
    dat[ticker] = merge.T.to_dict()
    
###############################################################################

with open('data\\5-cleaned_stock_data.json', 'w') as fp:
    json.dump(dat, fp)

###############################################################################
'''filter for 5 year data'''

ticker_list = []
for ticker in tqdm(dat.keys()):
    merge = pd.DataFrame(dat[ticker]).T
    if pd.to_datetime(merge.index[0]) <= datetime(2015,9,30,0,0):
        ticker_list.append(ticker)

ticker_name_list = json.loads(open('data\\1-ticker_name_list.json').read())

filtered_ticker_list = {}
for ticker in ticker_list:
    try:
        filtered_ticker_list[ticker] = ticker_name_list[ticker]
    except:
        print(ticker)

'''manually fixed with missing'''
#with open('data\\5-filtered_ticker_list.json', 'w') as fp:
#    json.dump(dat, fp)




