#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM with avg price 
@author: effie & Shan
"""
###############################################################################
'load api data'
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.cryptocurrencies import CryptoCurrencies

import gdelt
##########################################
import json
import urllib

import bs4 as bs
import requests
##########################################
'text modify'
import re
##########################################
import numpy as np
from tqdm import tqdm
import math
##########################################
from datetime import datetime
import time
##########################################
'to plot within notebook'
import matplotlib.pyplot as plt

##########################################
'importing required nn libraries'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  LSTM

################################################
import os
os.chdir('C:\\Users\\zhong\\Dropbox\\github\\sp500')
###############################################################################
"""load the name list of s&p 500 companies"""

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    company_names = []
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        ticker = "".join(re.findall("[a-zA-Z]+", ticker))
        tickers.append(ticker)
        company_names.append(row.findAll('td')[1].text)
        
    return tickers , company_names
##################################
    
tickers , company_names = save_sp500_tickers()

tickers[1:10]
company_names[1:10]

'actually 505 companies, as there are company like 	Alphabet Inc Class A and Alphabet Inc Class C'
len(company_names)

###############################################################################
'load adjusted s&p 500 data via alpha_vantage api'
'load bitcoin price data'
'load currency exchange rate data'
ts = TimeSeries(key='HGIKZ27XNJ7TS7N2')
fe = ForeignExchange(key='HGIKZ27XNJ7TS7N2')
cc = CryptoCurrencies(key='HGIKZ27XNJ7TS7N2')

'do not consider for adjust closing'
#SPX_data, meta_data = ts.get_daily_adjusted(symbol = 'SPX', outputsize = 'full')

data =  json.loads(open('sp500.json').read())

'load individual stock data for 505 companies on list'
for company in tqdm(tickers):
    max_date = np.max([datetime.strptime(item ,'%Y-%m-%d') for item in list(data.keys())])
    if company not in data[max_date.strftime('%Y-%m-%d')].keys():
        _data, _meta_data = ts.get_daily(symbol = company, outputsize = 'full')
        for date in _data.keys():
            if date in data.keys():        
                data[date][company]= float(_data[date]['4. close']) 
        time.sleep(12)


'save data'
with open('sp500.json', 'w') as fp:
    json.dump(data, fp)

data
