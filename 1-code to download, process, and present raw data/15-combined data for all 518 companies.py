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
import pandas as pd
import numpy as np
###############################################################################
ticker_sector = json.loads(open('data\\5-filtered_ticker_list.json').read())

_sector = []
for ticker in ticker_sector:
    _sector.append(ticker_sector[ticker]['sector'])
_sector = list(set(_sector))

stock_dat = json.loads(open('data\\5-cleaned_stock_data.json').read())
###############################################################################

_stock_ = pd.DataFrame()
for ticker in tqdm(ticker_sector):
    temp = pd.DataFrame(stock_dat[ticker]).T
    temp['adj_close'] = np.log(temp['Adj Close'])
    temp['date'] = pd.to_datetime(temp.index)
    temp = temp.groupby(pd.Grouper(key = 'date', freq = 'W-FRI')).mean().reset_index()
    
    temp = temp.sort_values('date')    
    temp['date'] = temp['date'].astype(str)
        
    temp["return_t"] = temp['adj_close'] - temp['adj_close'].shift()
    temp["return_t_plus_1"] = temp['adj_close'].shift(-1) - temp['adj_close']
    
    temp['sector'] = ticker_sector[ticker]['sector']
    temp['ticker'] = ticker    

    temp = temp[['return_t','return_t_plus_1','date','ticker','sector']]
    _stock_ = pd.concat([_stock_,temp], axis=0)

_stock_ = _stock_[pd.DatetimeIndex(_stock_['date']).year<2020]
_stock_ = _stock_[pd.DatetimeIndex(_stock_['date']).year>=2000]
_stock_ = _stock_.dropna()

###############################################################################

technical_indicators = json.loads(open('data\\6-technical_indicators.json').read())
_technical_indicators_ = pd.DataFrame()

for ticker in tqdm(ticker_sector):
    temp = pd.DataFrame(technical_indicators[ticker]) 
    temp['date'] = pd.to_datetime(temp.index)
    temp = temp.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()
    temp['date'] = temp['date'].astype(str)
    temp['ticker'] = ticker
    _technical_indicators_ = pd.concat([_technical_indicators_,temp], axis=0)

###############################################################################
fundamental_indices = json.loads(open('data\\7-fundamental_indices_data.json').read())
_fuundamental_indices_ = pd.DataFrame()

for ticker in tqdm(ticker_sector):
    temp = pd.DataFrame(fundamental_indices[ticker])
    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp[ pd.DatetimeIndex(temp['date']) > datetime(1999, 12, 1, 0, 0) ]
    
    temp = temp.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()
    temp['date'] = temp['date'].astype(str)
    temp['sector'] = ticker_sector[ticker]['sector']
    temp['ticker'] = ticker
    temp = temp[['MarketCap','PbRatio','PeRatio','PsRatio','date','sector','ticker']]
    _fuundamental_indices_ = pd.concat([_fuundamental_indices_,temp], axis=0)
    
###############################################################################
''' sentiment data by ticker'''

sentiment_data =  json.loads(open('data\\14-cleaned_all_500_company_news_sentiment.json').read())
sentiment = pd.DataFrame(sentiment_data)

sentiment['date'] = pd.to_datetime(sentiment['pub_date'])
sentiment = sentiment.sort_values('date')     
sentiment['sentiment'] = [item[0] - item[1] for item in list(sentiment['logit'])]

sentiment = sentiment[['ticker','date', 'sentiment']].groupby(['ticker',pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()
sentiment = sentiment[pd.DatetimeIndex(sentiment['date']).year<2020]
sentiment = sentiment[pd.DatetimeIndex(sentiment['date']).year>=2000]
sentiment['date'] = sentiment['date'].dt.date.astype(str)

###############################################################################

_dat_ = pd.merge(_stock_,_technical_indicators_,how = "left",on=['ticker','date'])
_dat_ = pd.merge(_dat_,_fuundamental_indices_,how = "left",on=['ticker','date'])
_dat_ = pd.merge(_dat_,sentiment,how = "left",on=['ticker','date'])

_dat_['sector'] = [ticker_sector[ticker]['sector'] for ticker in _dat_['ticker']]
_dat_['industry'] = [ticker_sector[ticker]['industry'] for ticker in _dat_['ticker']]

_dat_.to_csv("dat_518_companies.csv")

