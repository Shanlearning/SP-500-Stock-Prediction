'''
@author: zhong
'''

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

###############################################################################
import json
import stockstats

import pandas as pd
from tqdm import tqdm

###############################################################################

'''Stock.Indicators'''
'''https://github.com/DaveSkender/Stock.Indicators'''

stock_data_indices = json.loads(open('data\\5-cleaned_stock_data.json').read())

stock_indicators = {}
for ticker in tqdm(stock_data_indices.keys()):  
    value = pd.DataFrame.from_dict(stock_data_indices[ticker]).T
    temp = stockstats.StockDataFrame.retype(value)
    temp.index = pd.DatetimeIndex(temp.index)
    temp['clv'] = ( (temp['close']-temp['low']) - (temp['high']-temp['close']) )/(temp['high']-temp['low']) 
    temp['cmf'] = temp['clv'] * temp['volume']
    temp['cmf'] = temp['cmf'].rolling(window=20).sum()/temp['volume'].rolling(window=20).sum()
    temp['atr_percent'] = temp['atr'] / temp['close']
    
    temp.index = [date.strftime('%Y-%m-%d') for date in temp.index]
    
    stock_indicators[ticker] = temp[['adx','trix','adxr','cci','macd','macdh','rsi_14','kdjk','wr_14','atr_percent','atr','cmf']].dropna().to_dict()
    
with open('data\\6-technical_indicators.json', 'w') as fp:
    json.dump(stock_indicators, fp)