# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:03:21 2020

@author: zhong
"""
################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
# os.chdir('change to the mother working directory')

import bs4 as bs
import requests
import yfinance as yf

from tqdm import tqdm
import time
import pandas as pd
from datetime import datetime
import json

###############################################################################
''' get wiki ticker and name list '''

def save_sp500_information():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable', 'id': 'constituents'})
    tickers = {}   
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers[ticker] = {}  
        tickers[ticker]['company_names'] = row.findAll('td')[1].text.strip()
        tickers[ticker]['gics_sector'] = row.findAll('td')[3].text.strip()
        tickers[ticker]['gics_sub_industry'] = row.findAll('td')[4].text.strip()
        tickers[ticker]['date'] = row.findAll('td')[6].text.strip()
        
    table2 = soup.find('table', {'class': 'wikitable sortable', 'id': 'changes'})   
    removed_tickers = {}
    for row in table2.findAll('tr')[2:]:
        ticker = row.findAll('td')[3].text.strip()
        removed_tickers[ticker] = {} 
        removed_tickers[ticker]['date'] = row.findAll('td')[0].text.strip()
        removed_tickers[ticker]['company_names']= row.findAll('td')[4].text.strip()
        if len(row.findAll('td')) > 5:
            removed_tickers[ticker]['reason'] = row.findAll('td')[5].text.strip()
        else:
            removed_tickers[ticker]['reason'] = ""
    del removed_tickers['']
    return tickers , removed_tickers

wiki_tickers, wiki_removed_tickers = save_sp500_information()
###############################################################################

wiki_tickers = pd.DataFrame(wiki_tickers).T
wiki_removed_tickers = pd.DataFrame(wiki_removed_tickers).T

wiki_removed_tickers = wiki_removed_tickers[pd.DatetimeIndex(wiki_removed_tickers['date'])< datetime(2020,9,30,0,0)]
wiki_removed_tickers = wiki_removed_tickers[pd.DatetimeIndex(wiki_removed_tickers['date']).year>=2000]

###############################################################################
''' combine the name list '''

wiki_tickers = wiki_tickers.T.to_dict()
wiki_removed_tickers = wiki_removed_tickers.T.to_dict()

ticker_name_list = {}
for ticker in wiki_tickers:
    ticker_name_list[ticker] = {}
    ticker_name_list[ticker]['wiki_name'] = [wiki_tickers[ticker]['company_names']] 
for ticker in wiki_removed_tickers:
    if ticker in  wiki_tickers:
        ticker_name_list[ticker]['wiki_name'].append(wiki_removed_tickers[ticker]['company_names'])
    else:
        ticker_name_list[ticker] = {}
        ticker_name_list[ticker]['wiki_name'] = [wiki_removed_tickers[ticker]['company_names']]

''' regulate "." to "-" '''
for ticker in ticker_name_list:
    if "." in ticker:
        print(ticker)        
        
ticker_name_list["BRK-B"] = ticker_name_list.pop("BRK.B")
ticker_name_list["BF-B"] = ticker_name_list.pop("BF.B")

###############################################################################
''' quote data from yfinance '''

ticker_info_data = {}
for ticker in ticker_name_list:
    if ticker not in ticker_info_data:
        ticker_info_data[ticker] = {}

for ticker in tqdm( ticker_name_list ):
    if ticker_info_data[ticker] == {}:
        stock = yf.Ticker(ticker)
        try:
            ticker_info_data[ticker] = stock.info
        except:
            ticker_info_data[ticker] = {}
        time.sleep(10)

###############################################################################
''' add missed tickers and clean for yfinance data'''

want_keys = ['sector','industry','shortName','longName'] 

for ticker in ticker_info_data:
    if ticker_info_data[ticker] == {}:
        
        print(ticker)
    else:
        for _key in want_keys:
            if _key not in ticker_info_data[ticker]:
                print( _key + " " + ticker)

# manually fixing

ticker_info_data['GD']['sector'] = 'Industrials'
ticker_info_data['GD']['industry'] = 'Aerospace & Defense'

ticker_info_data['JCP']['sector'] = 'Consumer Cyclical'
ticker_info_data['JCP']['industry'] = 'Department Stores'

ticker_info_data['LM']['sector'] = 'Financial Services'
ticker_info_data['LM']['industry'] = 'Asset Management'

ticker_info_data['NE']['sector'] = 'Energy'
ticker_info_data['NE']['industry'] = 'Oil & Gas'

ticker_info_data["CHK"] = {'sector': 'Energy', 
                           'industry': 'Oil & Gas E&P',
                           'shortName': 'Chesapeake Energy Corporation',
                           'longName': 'Chesapeake Energy Corporation',
                           }        
        
ticker_info_data["DNR"] = {'sector': 'Energy', 
                           'industry': 'Oil & Gas E&P',
                           'shortName': 'Denbury Resources',
                           'longName': 'Denbury Resources Inc.',
                           }  

'''kust keep the information we want'''
for ticker in ticker_info_data:
    have_keys = list(set(ticker_info_data[ticker].keys()).intersection(set(want_keys)))
    ticker_info_data[ticker] = {_key: ticker_info_data[ticker][_key] for _key in have_keys }   

###############################################################################
''' merge the data again'''

for ticker in ticker_name_list:
    for item in ticker_info_data[ticker]:
        ticker_name_list[ticker][item] = ticker_info_data[ticker][item]

'''save data'''
with open('data\\1-ticker_name_list.json', 'w') as fp:
    json.dump(ticker_name_list, fp)









