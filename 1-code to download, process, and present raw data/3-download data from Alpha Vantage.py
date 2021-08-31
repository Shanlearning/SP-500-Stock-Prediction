# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 02:50:39 2020

@author: zhong
"""
###############################################################################
'load api data'
from alpha_vantage.timeseries import TimeSeries

##########################################
import json
import time
from tqdm import tqdm
##################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

_stock = json.loads(open('ticker_name_list.json').read())
tickers = list( _stock.keys() )

###############################################################################
ts = TimeSeries(key='a api key free to download at https://www.alphavantage.co/')

stock_data_AV = {}
#stock_data_AV  = json.loads(open('data\\stock_data_AV.json').read())

'load individual stock data for companies on list'
for stock in tqdm( tickers ):
    if stock not in stock_data_AV.keys():
        _data, _meta_data = ts.get_daily_adjusted(symbol = stock, outputsize = 'full')
        stock_data_AV[stock] = _data        
        time.sleep(11)

'save data'
with open('data\\stock_data_AV.json', 'w') as fp:
    json.dump(stock_data_AV, fp)



