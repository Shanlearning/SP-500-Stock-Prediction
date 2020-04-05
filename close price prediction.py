    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM with avg price 
@author: effie & Shan
"""
###############################################################################
'load data'
from alpha_vantage.timeseries import TimeSeries

###############################################################################
import numpy as np
from datetime import datetime
import math
###############################################################################
'to plot within notebook'
import matplotlib.pyplot as plt

###############################################################################
'importing required nn libraries'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  LSTM

###############################################################################
'load adjusted s&p 500 data'
'use adjusted close is better'
ts = TimeSeries(key='HGIKZ27XNJ7TS7N2')
# Get json object with the intraday data and another with  the call's metadata
data, meta_data = ts.get_daily_adjusted(symbol = 'SPX', outputsize = 'full')

Dates = [] 
dataset = []

'colse/adjusted close'
for date in data.keys():
    dataset.append( float(data[date]['4. close']) )
    Dates.append(datetime.strptime(date,'%Y-%m-%d'))



'scale data /log scaler'
def scale(data):
    data = [math.log(item) for item in data]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaleddata = scaler.fit_transform(np.asarray(dataset).reshape(-1,1))
    return(scaler,scaleddata)
    
    
scaler,scaled_data = scale(data=dataset)
scaled_data = scaler.fit_transform(np.asarray(dataset).reshape(-1,1))

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

#X_train = tf.convert_to_tensor(X, dtype=tf.float32)
#y_train = tf.convert_to_tensor(y, dtype=tf.float32)
#y_train = tf.reshape(y_train, shape = [len(y),1,1])

###############################################################################
'train/test test'


def split(X,y,split_rate=0.8):
    'X, y are from window function'
    train_size = int( len(X)*split_rate)
    
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    y = tf.reshape(y, shape = [len(y),1,1])

    X_train = X[0:train_size,:]
    y_train = y[0:train_size,:]
    X_test = X[train_size:,:]
    y_test = y[train_size:,:]
    
    return(X_train,y_train,X_test,y_test)

X_train,y_train,X_test,y_test= split(X,y,split_rate=0.8)


print("X_train size is {}".format(X_train.shape ))
print("y_train size is {}".format(y_train.shape ))
print("X_test size is {}".format(X_test.shape ))
print("y_test size is {}".format(y_test.shape ))

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



