import  os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')

###############################################################################
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd

###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
###############################################################################
'''trained using all stock data combined'''
original_val =  ['return_t','sentiment','cci','macdh','rsi_14','kdjk' ,'wr_14','cmf']
output_val = ['return_t_plus_1']

year_gap = 10
available_years = [item for item in range(2002,2020)]
_output = pd.DataFrame()
for year in tqdm(available_years):
    train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < year ].copy()
    train_df = train_df[pd.DatetimeIndex(train_df['date']).year >= (year-year_gap) ].copy()
    test_df = _dat_[ pd.DatetimeIndex(_dat_['date']).year == year ].copy()  
    
    _train_X = np.asarray(train_df[original_val])
    _train_y = np.asarray(train_df[output_val]).reshape(-1,)
    
    X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(_train_X)
    _train_X = X_transformer.transform(_train_X)
    _train_X  = np.nan_to_num(_train_X)
    
    _test_X =  np.asarray(test_df[original_val])
    _test_y =  np.asarray(test_df[output_val]).reshape(-1,)
    
    _test_X =  X_transformer.transform(np.asarray(test_df[original_val]))
    _test_X  = np.nan_to_num(_test_X)
    _test_X  = np.asarray(pd.DataFrame(_test_X).clip(-3.5,3.5))
    
    model = LinearRegression()      
    model.fit(_train_X,_train_y)
    _pred_y = model.predict(_test_X)
    test_df['LR_pred'] = _pred_y
    _output = pd.concat([_output,test_df], axis=0)

###############################################################################

'''testing data from 2003 to align with ensemble model'''
_output_2 = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('trained using all stock data combined')
print('MAE')
print( np.mean( np.abs(_output_2['LR_pred'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['LR_pred'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['LR_pred']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['LR_pred'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['LR_pred'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))

_output = _output[['date','ticker','LR_pred']].copy()
_output.to_csv("saved_output\\LR_pred.csv")
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
'''linear on sector level'''
###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv")
###############################################################################

original_val =  ['return_t','sentiment','cci','macdh','rsi_14','kdjk' ,'wr_14','cmf']
output_val = ['return_t_plus_1']

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
        train_df = train_df[pd.DatetimeIndex(train_df['date']).year >= (year-10) ].copy()
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
        
        model = LinearRegression()
        model.fit(_train_X,_train_y)
        _pred_y = model.predict(_test_X)
        
        test_df['LR_pred'] = _pred_y
        _output = pd.concat([_output,test_df], axis=0)
            
###############################################################################
_output = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('trained seperately on sector level')
print('MAE')
print( np.mean( np.abs(_output['LR_pred'] - _output['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output['LR_pred'] - _output['return_t_plus_1'])))
print('DA')
print(np.mean((_output['LR_pred']>=0) == (_output['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output['LR_pred'] >= 0),(_output['return_t_plus_1'] >= 0)) ) / sum(_output['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output['LR_pred'] < 0),(_output['return_t_plus_1'] < 0)) ) / sum(_output['return_t_plus_1'] < 0))

###############################################################################
'''linear on company level'''
###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv")
###############################################################################
available_years = [item for item in range(2002,2020)]
original_val =  ['return_t','cci','macdh','rsi_14','kdjk','wr_14','cmf','PeRatio', 'PsRatio', 'PbRatio']
output_val = ['return_t_plus_1']

yearly_report_RMSE = []
yearly_report_DA = []

_output = pd.DataFrame()
year_gap = 10     
for year in tqdm(available_years):
    train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < year ].copy()
    X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(train_df[original_val])
    
    _result = []
    _result_accuracy = []
    for ticker in ticker_sector:    
        temp = _dat_[_dat_['ticker'] == ticker].copy()
        train_df = temp[pd.DatetimeIndex(temp['date']).year < year ].copy()
        train_df = train_df[pd.DatetimeIndex(train_df['date']).year >= (year-year_gap) ].copy()
        test_df = temp[ pd.DatetimeIndex(temp['date']).year == year ].copy()  
        if len(train_df)>0:
            _train_X = np.asarray(train_df[original_val])
            _train_y = np.asarray(train_df[output_val]).reshape(-1,)
            
            _train_X = X_transformer.transform(_train_X)
            _train_X  = np.nan_to_num(_train_X)
            _train_X  = np.asarray(pd.DataFrame(_train_X).clip(-4.5,4.5))
            
            _test_X =  np.asarray(test_df[original_val])
            _test_y =  np.asarray(test_df[output_val]).reshape(-1,)
            
            _test_X =  X_transformer.transform(np.asarray(test_df[original_val]))
            _test_X  = np.nan_to_num(_test_X)
            _test_X  = np.asarray(pd.DataFrame(_test_X).clip(-4.5,4.5))
           
            model = LinearRegression()
            model.fit(_train_X,_train_y)
            _pred_y = model.predict(_test_X)
            
            test_df['LR_pred'] = _pred_y
            _output = pd.concat([_output,test_df], axis=0)
               
###########################################################################
_output = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('trained seperately on each individual stock')
print('MAE')
print( np.mean( np.abs(_output['LR_pred'] - _output['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output['LR_pred'] - _output['return_t_plus_1'])))
print('DA')
print(np.mean((_output['LR_pred']>=0) == (_output['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output['LR_pred'] >= 0),(_output['return_t_plus_1'] >= 0)) ) / sum(_output['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output['LR_pred'] < 0),(_output['return_t_plus_1'] < 0)) ) / sum(_output['return_t_plus_1'] < 0))


