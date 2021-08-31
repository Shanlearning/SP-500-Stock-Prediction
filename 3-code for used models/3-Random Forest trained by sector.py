################################
import  os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')
###############################################################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
#import _pickle as cPickle
###############################################################################
ticker_sector = json.loads(open('2-cleaned_data\\ticker_sector_information.json').read())
_sector = []
for ticker in ticker_sector:
    _sector.append(ticker_sector[ticker]['sector'])
_sector = list(set(_sector))

_industry = []
for ticker in ticker_sector:
    _industry.append(ticker_sector[ticker]['industry'])
_industry = list(set(_industry))
###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
###############################################################################
original_val =  ['return_t','sentiment','PeRatio', 'PsRatio', 'PbRatio','cci','macd','rsi_14','kdjk' ,'wr_14','atr_percent','cmf']
output_val = ['return_t_plus_1']
###############################################################################

available_years = [item for item in range(2002,2020)]

_output = pd.DataFrame()
for year in tqdm(available_years):   
    _result = []
    _result_accuracy = []
    train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < year ].copy()
    X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(np.asarray(train_df[original_val]))
    
    for sector in _sector:    
        temp = _dat_[_dat_['sector'] == sector].copy()       
        train_df = temp[pd.DatetimeIndex(temp['date']).year < year ].copy()
        train_df = train_df[pd.DatetimeIndex(train_df['date']).year >= (year-2) ].copy()
        test_df = temp[ pd.DatetimeIndex(temp['date']).year == year ].copy()  
        
        _train_X = np.asarray(train_df[original_val])
        _train_y = np.asarray(train_df[output_val]).reshape(-1,)
          
        _train_X = X_transformer.transform(_train_X)
        _train_X  = np.nan_to_num(_train_X)
       
        _test_X =  np.asarray(test_df[original_val])
        _test_y =  np.asarray(test_df[output_val]).reshape(-1,)
        
        _test_X =  X_transformer.transform(np.asarray(test_df[original_val]))
        _test_X  = np.nan_to_num(_test_X)
        _test_X  = np.asarray(pd.DataFrame(_test_X).clip(-4.5,4.5))
        
        model = RandomForestRegressor(n_estimators = 20, criterion = 'mae', max_depth=8, max_features="auto", n_jobs = -1)       
        model.fit(_train_X,_train_y)
        _pred_y = model.predict(_test_X)
        
        _result.extend( np.square( (_test_y - _pred_y ) ) ) 
        _result_accuracy.extend(  (_pred_y>=0) == (_test_y>=0) ) 
        
        #with open('saved_models\\'+str(year) + '_' + str(sector)+ '.pkl', 'wb') as fid:
        #    cPickle.dump(model, fid)   
        test_df['RF_pred'] = model.predict(_test_X) 
        _output = pd.concat([_output,test_df], axis=0)
        print(np.mean((_output['RF_pred']>=0) == (_output['return_t_plus_1']>=0)))
        
###############################################################################
_output_2 = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('Random Forest trained separately by each sector')
print('MAE')
print( np.mean( np.abs(_output_2['RF_pred'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['RF_pred'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['RF_pred']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['RF_pred'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['RF_pred'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))

#_output = _output[['date','ticker','RF_pred']].copy()
#_output.to_csv("saved_output\\RF_pred.csv")
