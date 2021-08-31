import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

###############################################################################
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

import stockstats
###############################################################################
ticker_sector = json.loads(open('data\\5-filtered_ticker_list.json').read())

###############################################################################
'''sp 500 index data download manually from Yahoo Finance'''
stock_dat = pd.read_excel("data\\16-sp500_index_data.xlsx",sheet_name="Sheet1")
stock_dat.columns  = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
stock_dat['date'] = pd.to_datetime(stock_dat['date'])
stock_dat = stock_dat.sort_values('date')    

temp = stockstats.StockDataFrame.retype(stock_dat.copy())

temp.index = pd.DatetimeIndex(temp.index)
temp.index = [date.strftime('%Y-%m-%d') for date in temp.index]
temp['clv'] = ( (temp['close']-temp['low']) - (temp['high']-temp['close']) )/(temp['high']-temp['low']) 
temp['cmf'] = temp['clv'] * temp['volume']
temp['cmf'] = temp['cmf'].rolling(window=20).sum()/temp['volume'].rolling(window=20).sum()

temp = temp[['adj close','trix','adxr','close','adx','cci',"macdh",'rsi_14','kdjk','wr_14','atr','cmf']]

temp['atr_percent'] = temp['atr'] / temp['close']
temp["return_t"] = temp['adj close'] - temp['adj close'].shift()
temp = temp[['adx','trix','adxr','cci','macdh','rsi_14','kdjk','wr_14','atr_percent','atr','cmf']]
temp = pd.DataFrame(temp)
temp['date'] = pd.to_datetime(temp.index)

temp = temp.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).mean().reset_index()
temp['date'] = temp['date'].astype(str)

tech_indices = temp.copy()
tech_indices = tech_indices[pd.DatetimeIndex(tech_indices['date']).year<2020]
tech_indices = tech_indices[pd.DatetimeIndex(tech_indices['date']).year>=2000]

###############################################################################

stock_dat = pd.read_excel("data\\16-sp500_index_data.xlsx",sheet_name="Sheet1")
stock_dat.columns  = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
stock_dat['date'] = pd.to_datetime(stock_dat['date'])
stock_dat = stock_dat.sort_values('date')    
stock_dat['adj close'] = np.log(stock_dat['adj close'] )
stock_dat = stock_dat.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')])[['adj close']].mean().reset_index()
stock_dat = stock_dat.sort_values('date')    
stock_dat['date'] = stock_dat['date'].astype(str)

stock_dat["return_t"] = stock_dat['adj close'] - stock_dat['adj close'].shift()
stock_dat["return_t_plus_1"] = stock_dat['adj close'].shift(-1) - stock_dat['adj close']
stock_dat = stock_dat[pd.DatetimeIndex(stock_dat['date']).year<2020]
stock_dat = stock_dat[pd.DatetimeIndex(stock_dat['date']).year>=2000]
stock_dat = stock_dat.dropna()

###############################################################################
'''using the median of individual companies values for the fundamental indices data'''

fundamental_indices = json.loads(open('data\\7-fundamental_indices_data.json').read())
_fuundamental_indices_ = pd.DataFrame()

for ticker in tqdm(ticker_sector):
    temp = pd.DataFrame(fundamental_indices[ticker])
    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp[ pd.DatetimeIndex(temp['date']) > datetime(1999, 1, 1, 0, 0) ]
    temp['sector'] = ticker_sector[ticker]['sector']
    temp['ticker'] = ticker
    temp = temp[['MarketCap','PbRatio','PeRatio','PsRatio','date','sector','ticker']]
    _fuundamental_indices_ = pd.concat([_fuundamental_indices_,temp], axis=0)
    
_fuundamental_indices_ = _fuundamental_indices_.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')])[['PbRatio', 'PeRatio', 'PsRatio']].median().reset_index()
_fuundamental_indices_['date'] = _fuundamental_indices_['date'].astype(str)

_fuundamental_indices_ = _fuundamental_indices_[pd.DatetimeIndex(_fuundamental_indices_['date']).year<2020]
_fuundamental_indices_ = _fuundamental_indices_[pd.DatetimeIndex(_fuundamental_indices_['date']).year>=2000]
_fuundamental_indices_ = _fuundamental_indices_.dropna()

###############################################################################
'''use both the median news of individual companies, as well the news directly about sp500 index'''

spsentiment = json.loads(open('data\\14-cleaned_sp_500_index_sentiment.json').read())
spsentiment = pd.DataFrame(spsentiment)
spsentiment['date'] = pd.to_datetime(spsentiment['pub_date'])
spsentiment = spsentiment.sort_values('date')     
spsentiment['spsentiment'] = [item[0] - item[1] for item in list(spsentiment['logit'])]

spsentiment = spsentiment[['date', 'spsentiment']].groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()

spsentiment = spsentiment[pd.DatetimeIndex(spsentiment['date']).year<2020]
spsentiment = spsentiment[pd.DatetimeIndex(spsentiment['date']).year>=2000]
spsentiment['date'] = spsentiment['date'].dt.date.astype(str)

sentiment = json.loads(open('data\\14-cleaned_all_500_company_news_sentiment.json').read())
sentiment = pd.DataFrame(sentiment)
sentiment['date'] = pd.to_datetime(sentiment['pub_date'])
sentiment = sentiment.sort_values('date')     
sentiment['sentiment'] = [item[0] - item[1] for item in list(sentiment['logit'])]
sentiment = sentiment[['date', 'sentiment']].groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()

sentiment = sentiment[pd.DatetimeIndex(sentiment['date']).year<2020]
sentiment = sentiment[pd.DatetimeIndex(sentiment['date']).year>=2000]
sentiment['date'] = sentiment['date'].dt.date.astype(str)
###############################################################################

_dat_ = pd.merge(stock_dat,tech_indices,how = "left",on=['date'])
_dat_ = pd.merge(_dat_,_fuundamental_indices_,how = "left",on=['date'])
_dat_ = pd.merge(_dat_,spsentiment,how = "left",on=['date'])
_dat_ = pd.merge(_dat_,sentiment,how = "left",on=['date'])

_dat_.to_csv("dat_sp500_index.csv")

