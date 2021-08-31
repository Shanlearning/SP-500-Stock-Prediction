# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 00:06:53 2020

@author: zhong
"""

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
# os.chdir('change to the mother working directory')'

import yfinance as yf
from tqdm import tqdm
import time
import json

ticker_name_list = json.loads(open('data\\ticker_name_list.json').read())
tickers = list(ticker_name_list.keys())

stock_data_YF = {}
for ticker in tqdm( tickers ):
    stock = yf.Ticker(ticker)
    _closing_dat = stock.history(auto_adjust=False,back_adjust=False,rounding=False,period="max")   
    stock_data_YF[ticker] = _closing_dat.T.to_dict()
    time.sleep(10)
    
with open('data\\stock_data_YF.json', 'w') as fp:
    json.dump(stock_data_YF, fp)



