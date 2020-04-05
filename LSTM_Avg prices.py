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
import numpy as np
import math
##########################################
from datetime import datetime
##########################################
'to plot within notebook'
import matplotlib.pyplot as plt

##########################################
'importing required nn libraries'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  LSTM

###############################################################################
"""load the name list of s&p 500 companies"""

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    company_names = []
    tickers = []
    for row in table.findAll('tr')[1:]:
        tickers.append(row.findAll('td')[0].text.strip())
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

ts = TimeSeries(key='HGIKZ27XNJ7TS7N2')
fe = ForeignExchange(key='HGIKZ27XNJ7TS7N2')
cc = CryptoCurrencies(key='HGIKZ27XNJ7TS7N2')

data, meta_data = ts.get_daily_adjusted(symbol = 'SPX', outputsize = 'full')

Dates = [] 
SPX_data = []

for date in data.keys():
    SPX_data.append( float(data[date]['4. close']) )
    Dates.append(datetime.strptime(date,'%Y-%m-%d'))
    
Dates[0].strftime('%Y-%m-%d')

data, _meta_data = ts.get_daily_adjusted(symbol = tickers[0], outputsize = 'full')

cc.get_digital_currency_daily(symbol='BTC', market='USD')
fe.get_currency_exchange_daily(from_symbol = 'USD', to_symbol = 'CNY')


    
'推荐你用log scaler'
dataset = [math.log(item) for item in dataset]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.asarray(dataset).reshape(-1,1))

###############################################################################
'sentiment data'
###############################################################################

'load news data'
gd2 = gdelt.gdelt(version=2)
# Single 15 minute interval pull, output to json format with mentions table
events = gd2.Search(['2015 Feb 18',datetime.today().strftime('%Y %b %d')],translation = False, table='events',output='pandas dataframe')


np.unique([str(item) for item in events.Actor1Name])[1:40]
events.Actor2Name
events.CAMEOCodeDescription
'This is the average “tone” of a event, by transfering words into sentiment scores'
events.AvgTone

"""Python client calling Knowledge Graph Search API"""
api_key = 'AIzaSyByheCFHN9ybO53IbJ-TZiKOh8Bzl885VQ'
query = 'WRK'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
params = {
    'query': query,
    'limit': 10,
    'indent': True,
    'key': api_key,
}
url = service_url + '?' + urllib.parse.urlencode(params)
response = json.loads(urllib.request.urlopen(url).read())
for element in response['itemListElement']:
  print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')
###############################################################################
'process data'
scaled_data.shape
window_size=100
def window_data(data, window_size):
    X= [ ]
    y= [ ]
    
    i=0
    while (i + window_size)<=len(data)-1:
        X.append(data[i: i+ window_size ]  )
        y.append(data[i+window_size])
        i += 1
    assert len(X) == len(y)
    return X,y

X,y = window_data(scaled_data, window_size )

X_train = tf.convert_to_tensor(X, dtype=tf.float32)
y_train = tf.convert_to_tensor(y, dtype=tf.float32)
y_train = tf.reshape(y_train, shape = [len(y),1,1])

###############################################################################
'train test'
'do it yourself! 自己做！'
train_size = int( len(dataset)*0.8)
#train = dataset[0:train_size,:]
#test = dataset[train_size:,:]

###############################################################################
'model 1'
###############################################################################
'create and fit the LSTM network'
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size,1)))
model.add(Dense(1)) 
model.add(tf.keras.layers.GlobalAveragePooling1D())


model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
###############################################################################
'fit model'
model.fit(X_train, y_train, epochs=20, batch_size=800)

model.predict(X_train).shape

'transfer back'
y_pred = [math.exp(item) for item in  scaler.inverse_transform(model.predict(X_train)) ]
y_true = [math.exp(item) for item in  scaler.inverse_transform(tf.reshape(y_train, [-1,1])) ]

###############################################################################
'plots'
'on log scale'
plt.plot(Dates[window_size:],scaler.inverse_transform(model.predict(X_train)))
plt.plot(Dates[window_size:],scaler.inverse_transform(tf.reshape(y_train, [-1,1])))

###############################################################################
'direct value'
plt.plot(Dates[window_size:],y_pred)
plt.plot(Dates[window_size:],y_true)

