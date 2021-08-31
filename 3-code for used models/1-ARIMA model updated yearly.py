# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 07:23:02 2020

@author: zhong
"""
################################
import  os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')
###############################################################################
import json

import pandas as pd
import pmdarima as pm
from pmdarima.arima import ndiffs
import numpy as np
from tqdm import tqdm

###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
ticker_sector = json.loads(open('2-cleaned_data\\ticker_sector_information.json').read())
###############################################################################

available_years = [item for item in range(2003,2020)]

_output = pd.DataFrame()
for _year in tqdm(available_years):
    
    train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < _year ].copy()
    test_df = _dat_[ pd.DatetimeIndex(_dat_['date']).year == _year ].copy()  
    
    for ticker in tqdm(ticker_sector, leave=False):
        
        temp = train_df[train_df['ticker'] == ticker].copy()
        if len(temp) > 20:
            temp['date'] = pd.to_datetime(temp['date'])   
            temp = temp.sort_values('date') 
            temp_test = test_df[test_df['ticker'] == ticker].copy()
            temp_test['date'] = pd.to_datetime(temp_test['date'])   
            temp_test = temp_test.sort_values('date') 
            try:   
                _return_dat = np.concatenate( [np.asarray(temp['return_t'].iloc[0]) ,np.asarray(temp['return_t_plus_1'])], axis = None) 
                 
                kpss_diffs = ndiffs(_return_dat, alpha=0.05, test='kpss', max_d=4)
                adf_diffs = ndiffs(_return_dat, alpha=0.05, test='adf', max_d=4)
                n_diffs = max(adf_diffs, kpss_diffs)
                
                model = pm.auto_arima(_return_dat, start_p=1, start_q=1, d=n_diffs, seasonal=False, stepwise=True,
                                     suppress_warnings=True, error_action="ignore", max_p=4, information_criterion = "aic", 
                                     max_q=4, trace= False )
                            
                _test_y = np.asarray(temp_test['return_t_plus_1']) 
                _pred_y = []
                
                for pos in _test_y:
                    _pred_y.append(model.predict(1)[0])
                    model.update(pos)
            except:    
                _return_dat = np.asarray(temp['return_t']) 
    
                kpss_diffs = ndiffs(_return_dat, alpha=0.05, test='kpss', max_d=4)
                adf_diffs = ndiffs(_return_dat, alpha=0.05, test='adf', max_d=4)
                n_diffs = max(adf_diffs, kpss_diffs)
                
                model = pm.auto_arima(_return_dat, start_p=1, start_q=1, d=n_diffs, seasonal=False, stepwise=True,
                                     suppress_warnings=True, error_action="ignore", max_p=4, information_criterion = "aic", 
                                     max_q=4, trace= False )
    
                _test_y = np.concatenate( [np.asarray(temp_test['return_t'].iloc[0]) ,np.asarray(temp_test['return_t_plus_1'])], axis = None) 
                _pred_y = []
                
                for pos in _test_y:
                    _pred_y.append(model.predict(1)[0])
                    model.update(pos)
                _test_y = np.asarray(temp_test['return_t_plus_1']) 
                del _pred_y[0] 
                    
            _pred_y = np.asarray(_pred_y)
            
            temp_test['ARIMA_pred'] = _pred_y
            _output = pd.concat([_output,temp_test], axis=0)
            print(np.mean((_output['ARIMA_pred']>=0) == (_output['return_t_plus_1']>=0)))
            
###############################################################################
_output_2 = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('ARIMA yearly update')
print('MAE')
print( np.mean( np.abs(_output_2['ARIMA_pred'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['ARIMA_pred'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['ARIMA_pred']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['ARIMA_pred'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['ARIMA_pred'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))

_output = _output[['date','ticker','ARIMA_pred']].copy()
_output['date'] = pd.DatetimeIndex(_output['date']).astype(str)
_output.to_csv("saved_output\\ARIMA.csv")