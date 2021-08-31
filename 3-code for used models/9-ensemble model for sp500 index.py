import os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')
###############################################################################
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import pandas as pd

###############################################################################
_sp_dat_ = pd.read_csv("2-cleaned_data\\dat_sp500_index.csv",index_col = 0)
_sp_dat_['date'] = pd.to_datetime(_sp_dat_['date'])
_sp_dat_ = _sp_dat_[['date','return_t_plus_1']].copy()

###############################################################################

_output = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
_output = _output[['date','ticker']].copy()

_LR_output = pd.read_csv("saved_output\\LR.csv",index_col = 0) 
_RF_output = pd.read_csv("saved_output\\RF_sec.csv",index_col = 0) 
_Dense_output = pd.read_csv("saved_output\\Dense_monthly_update_10_year.csv",index_col = 0) 
_LSTM_two_layer = pd.read_csv("saved_output\\LSTM_pred_two_layer_mae_linear_activation_length_3.csv",index_col = 0) 
_LSTM_stock_output = pd.read_csv("saved_output\\LSTM_pred_one_layer_mae_linear_activation_length_3_stock.csv",index_col = 0) 
###############################################################################

_output = pd.merge(_output,_LR_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_RF_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_Dense_output,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_LSTM_two_layer,how = "left",on=['date','ticker'])
_output = pd.merge(_output,_LSTM_stock_output,how = "left",on=['date','ticker'])
###############################################################################

_output['date'] = pd.to_datetime(_output['date'])
_output.isna().sum()
_output = _output.dropna().copy()
_output = _output.groupby([pd.Grouper(key = 'date', freq = 'W-FRI')]).median().reset_index()

_sp_dat_ = pd.merge(_sp_dat_,_output,how = "left",on=['date'])
_sp_dat_ = _sp_dat_[pd.DatetimeIndex(_sp_dat_['date']).year>=2002].copy()

###############################################################################

available_years = [item for item in range(2003,2020)]

vals = ['LR_pred','RF_pred','LSTM_two_layer_length_3_pred','LSTM_pred_stock','Dense_monthly_update_10_year']

_output_2 = pd.DataFrame()
for _year in tqdm(available_years):
    temp = _sp_dat_[pd.DatetimeIndex(_sp_dat_['date']).year < _year ].copy()
    temp = temp[pd.DatetimeIndex(temp['date']).year >= (_year-1) ].copy()
    _train_X = np.asarray(temp[vals])
    _train_X  = np.nan_to_num(_train_X)
    _train_y = np.asarray(temp['return_t_plus_1']).reshape(-1,)
    test = _sp_dat_[pd.DatetimeIndex(_sp_dat_['date']).year == _year ].copy()
    _test_X = np.asarray(test[vals])
    _test_X  = np.nan_to_num(_test_X)
    _test_y = np.asarray(test['return_t_plus_1']).reshape(-1,)
    
    model = LinearRegression(fit_intercept = False, positive = True)      
    model.fit(_train_X,_train_y)
    _pred_y = model.predict(_test_X)
    test['ensemble'] = _pred_y
    _output_2 = pd.concat([_output_2,test], axis=0)

print('Ensemble model for the sp500 index')
print('MAE')
print( np.mean( np.abs(_output_2['ensemble'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['ensemble'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['ensemble']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['ensemble'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['ensemble'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))
