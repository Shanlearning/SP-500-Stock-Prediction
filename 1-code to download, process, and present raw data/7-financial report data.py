# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 02:25:46 2020

@author: zhong
"""
################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')
###############################################################################
import json
import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime,timedelta
import numpy as np


_stock_dat = stock_data_indices = json.loads(open('data\\5-cleaned_stock_data.json').read())


###############################################################################
'''fundamential indice data was downloaded manually'''
'''loaded together by repeatedly read csv files'''
tickers_list = list(_stock_dat.keys())

'''first load all variable names from csv files'''
Lst = []
for ticker in tqdm(tickers_list):
    _valuation_measures = pd.read_csv( 'data\\7-merged_finance_data\\' + str(ticker) + "\\" + str(ticker) +'_monthly_valuation_measures.csv',
                                 index_col=0)
    if 'ttm' in _valuation_measures.columns:
        del _valuation_measures['ttm']

    Lst.extend( list( _valuation_measures.index ) )

val_names = list( set(Lst) )
val_names.sort(reverse=False)
val_names

###############################################################################
'''date formats'''
def _sort_fun(x):
    return datetime.strptime(x, '%m/%d/%Y')

'''fill na value and fix comma issue'''
def exclude_comma_transfer_float(_data,name):
    temp = []
    for item in list (_data.loc[name].values ):
        if pd.isnull(item):
            temp.append(float('nan'))
        else:
            temp.append( float( re.sub(',','',item) ) )
    return temp
###############################################################################
'''reconstruction the sequence to include every date'''
sdate = datetime(1995,11,1) # start date
edate = datetime(2020,9,30)
delta = edate - sdate       # as timedelta
available_dates= []
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i) 
    available_dates.append(day.strftime('%Y-%m-%d')) 
    
def extract_daily_statement(_valuation_measures,available_dates,val_names):
    temp = pd.DataFrame(index=pd.to_datetime(available_dates), columns = val_names)
    for dates in _valuation_measures.keys():
        if dates in available_dates:
            temp.iloc[available_dates.index(dates)] = [_valuation_measures[dates][val] for val in val_names]
    temp['date'] = temp.index.astype(str)
    return temp

###############################################################################

dat = {}
for ticker in tqdm(tickers_list):
    _valuation_measures = pd.read_csv( 'data\\7-merged_finance_data\\' + str(ticker) + "\\" + str(ticker) +'_monthly_valuation_measures.csv',
                                 index_col=0)
    if 'ttm' in _valuation_measures.columns:
        del _valuation_measures['ttm']
    
    for val in val_names:
        if val not in _valuation_measures.index:
            _valuation_measures.loc[val] = float('nan')
    
    '''fix the order of variable name and dates'''
    _valuation_measures = _valuation_measures.reindex(sorted(_valuation_measures.index), axis=0)
    _valuation_measures = _valuation_measures.reindex(sorted(_valuation_measures.columns,key =_sort_fun), axis=1)
    _valuation_measures.columns = [ datetime.strptime(item,
                            '%m/%d/%Y').strftime('%Y-%m-%d') for item in _valuation_measures.columns]
    
    '''fill na value and fix comma issue'''
    for name in val_names:
        _valuation_measures.loc[name] = exclude_comma_transfer_float(_valuation_measures,name)
    _valuation_measures = _valuation_measures.to_dict()
    
    '''load stock data to help adjust finance index to daily level'''
    temp_stock = pd.DataFrame(_stock_dat[ticker]).T
    temp_stock['date'] = temp_stock.index.astype(str)
    
    '''reconstruct the fundamental indice data to daily level'''
    temp_index = extract_daily_statement(_valuation_measures,available_dates,val_names)   
    '''fill in the stock values first'''
    temp_index = pd.merge(temp_index,temp_stock,how = "left",on=['date'])
    temp_index['Adj Close'] = temp_index['Adj Close'].fillna(method='ffill') 
    temp_index['Adj Close'] = temp_index['Adj Close'].fillna(method='bfill') 
    temp_index["return_t"] = np.log(temp_index['Adj Close']) - np.log(temp_index['Adj Close'].shift())
    
    '''adjust the values by return'''
    for val in val_names:
        if (~temp_index[val].isna()).sum() != 0:
            mid = [i for i,v in enumerate(temp_index[val]) if not np.isnan(v)][0]
            for i in range(mid,len(temp_index[val])):
                if np.isnan(temp_index.iloc[i][val]):
                   temp_index.loc[i,val] = temp_index.iloc[i-1][val] * np.exp(temp_index.iloc[i]["return_t"])
    
    '''keep trading days part of data'''
    temp = pd.merge(temp_stock,temp_index,how = "left",on=['date'])
    temp = temp[val_names+['date']]
    
    dat[ticker] = temp.to_dict()

'save data'
with open('data\\fundamental_indices_data.json', 'w') as fp:
    json.dump(dat, fp)    

    

    
