import  os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')

###############################################################################
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import pandas as pd

###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
###############################################################################

original_val =  ['return_t','sentiment','PeRatio', 'PsRatio', 'PbRatio','cci','macd','rsi_14','kdjk' ,'wr_14','atr_percent','cmf']
output_val = ['return_t_plus_1']

###############################################################################
pd.set_option('display.max_columns', None)

_dat_['date'] = pd.to_datetime(_dat_['date'])

available_years = [item for item in range(2002,2020)]
_result = []
_result_accuracy = []
_output_linear_all = pd.DataFrame()

def set_model():
    _input = tf.keras.Input(shape=(len(original_val)),dtype=tf.float32,name='stock_dense_input')
    dense_1 = Dense(units=48, activation='relu')(_input)
    dropout_1 = Dropout(0.6)(dense_1)
    result = Dense(units= 1,kernel_initializer= tf.initializers.zeros(), activation=None)(dropout_1)  
    model = tf.keras.Model(inputs = _input,
                          outputs = result,
                          name='model1')
    model.compile(loss='mean_absolute_error', optimizer='adam')    
    return model

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    mode='min')

MAX_EPOCHS = 3500
model = set_model()
model.summary()
model.save_weights('saved_models\\initial_weights.h5')
model.load_weights('saved_models\\initial_weights.h5')

_output = pd.DataFrame()
for year in tqdm(available_years):
    for month in tqdm(range(1,13),leave=False, disable = True):
        train_df = _dat_[pd.DatetimeIndex(_dat_['date']) < str(year)+'-'+str(month)+'-01' ].copy()
        train_df = train_df[pd.DatetimeIndex(train_df['date']) >= str(year-10)+'-'+str(month)+'-01'].copy()
        X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(train_df[original_val])
        train_X = np.nan_to_num(X_transformer.transform(train_df[original_val]))
        train_df[original_val] = np.asarray(pd.DataFrame(train_X))
        
        test_df = _dat_[ pd.DatetimeIndex(_dat_['date']).year == year].copy()  
        test_df = test_df[ pd.DatetimeIndex(test_df['date']).month == month].copy() 
        test_X = np.nan_to_num(X_transformer.transform(test_df[original_val]))
        test_df[original_val] = np.asarray(pd.DataFrame(test_X).clip(-4.5,4.5))
            
        train_index,test_index = train_test_split(range(0,len(train_df)), test_size=0.1)
          
        _train_X = np.asarray(train_df[original_val].iloc[train_index]  )
        _train_y = np.asarray(train_df[output_val].iloc[train_index])
        
        _validate_X = np.asarray(train_df[original_val].iloc[test_index]  )
        _validate_y = np.asarray(train_df[output_val].iloc[test_index])
        
        _train = tf.data.Dataset.from_tensor_slices((_train_X,_train_y)).batch(2048)
        _validate = tf.data.Dataset.from_tensor_slices((_validate_X,_validate_y)).batch(2048)
        
        _test_X =  np.asarray(test_df[original_val])
        _test_y =  np.asarray(test_df[output_val])
        
        model.load_weights('saved_models\\initial_weights.h5')
        history = model.fit(_train, epochs=MAX_EPOCHS, validation_data = _validate,verbose=0,
                callbacks=[early_stopping])
        _pred_y = model.predict(_test_X)
    
        test_df['Dense_monthly_update_10_year'] = _pred_y 
        _output = pd.concat([_output,test_df], axis=0)
    
    print(np.mean((_output['Dense_monthly_update_10_year']>=0) == (_output['return_t_plus_1']>=0)))

###############################################################################

_output_2 = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('FFNN monthly update 10 years rolling')
print('MAE')
print( np.mean( np.abs(_output_2['Dense_monthly_update_10_year'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['Dense_monthly_update_10_year'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['Dense_monthly_update_10_year']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['Dense_monthly_update_10_year'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['Dense_monthly_update_10_year'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))

#_output = _output[['date','ticker','Dense_monthly_update_10_year']].copy()
#_output.to_csv("saved_output\\Dense_monthly_update_10_year.csv")

