import  os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')
###############################################################################
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
import json

from tqdm import tqdm
import numpy as np
import pandas as pd

###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_sp500_index.csv",index_col = 0)
ticker_sector = json.loads(open('2-cleaned_data\\ticker_sector_information.json').read())
###############################################################################

available_years = [item for item in range(2003,2020)]

original_val =  ['return_t','spsentiment','PeRatio', 'PsRatio', 'PbRatio','cci','macdh','rsi_14','kdjk' ,'wr_14','atr_percent','cmf']
output_val = ['return_t_plus_1']

_output = pd.DataFrame()
for year in tqdm(available_years):
    train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < year ].copy()
    train_df = train_df[pd.DatetimeIndex(train_df['date']).year >= (year-10) ].copy()
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
    _test_X  = np.asarray(pd.DataFrame(_test_X).clip(-4.5,4.5))
    
    model = LinearRegression()
    model.fit(_train_X,_train_y)
    _pred_y = model.predict(_test_X)
    test_df['LR_pred'] = _pred_y
    
    model = RandomForestRegressor(n_estimators = 4000,criterion = 'mae', max_depth=8, max_features= 8)      
    model.fit(_train_X,_train_y)
    _pred_y = model.predict(_test_X)
    test_df['RF_pred'] = _pred_y
    _output = pd.concat([_output,test_df], axis=0)
    
print('DA of Linear Regression for sp500 index')
print(np.mean((_output['LR_pred']>=0) == (_output['return_t_plus_1']>=0))   )
print('DA of Random Forest for sp500 index')
print(np.mean((_output['RF_pred']>=0) == (_output['return_t_plus_1']>=0))   )