# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:40:46 2020

@author: zhong
"""
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
###############################################################################
'load s&p 500 data'
ts = TimeSeries(key='HGIKZ27XNJ7TS7N2')
# Get json object with the intraday data and another with  the call's metadata
data, meta_data = ts.get_intraday(symbol = 'SPX', interval = '15min' , outputsize = 'full')


###############################################################################
import gdelt
from datetime import datetime
'load news data'
gd2 = gdelt.gdelt(version=2)
# Single 15 minute interval pull, output to json format with mentions table
events = gd2.Search(['2015 Feb 18',datetime.today().strftime('%Y %b %d')],translation = False, table='events',output='pandas dataframe')


event_dates = [datetime.strptime(str(item),'%Y%m%d%H%M%S') for item in events.DATEADDED]

event_dates

print(datetime.strptime(str(events.DATEADDED[0]),'%Y%m%d%H%M%S'))

events.keys()

a = np.unique([str(item) for item in events.Actor2Name])

events.Actor1Name
np.unique(events.Actor2Name)

'This is the average “tone” of a event, by transfering words into sentiment scores'
events.AvgTone


events.keys()

import numpy as np
np.unique(events.CAMEOCodeDescription[0])


len(a)


events.columns











