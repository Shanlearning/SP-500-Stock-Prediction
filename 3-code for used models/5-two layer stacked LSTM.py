################################
import  os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')

###############################################################################
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
ticker_sector = json.loads(open('2-cleaned_data\\ticker_sector_information.json').read())
###############################################################################
original_val =  ['return_t','sentiment','PeRatio','PsRatio','PbRatio','cci','macdh','rsi_14','kdjk' ,'wr_14','atr_percent','cmf']
output_val = ['return_t_plus_1']

###############################################################################
input_width=3;
input_slice = slice(0, input_width)

available_years = [item for item in range(2002,2020)]
val_len = len(original_val)+len(output_val)

def split_window(features):
        inputs = features[:,input_slice,:]
        inputs = tf.stack(
                [inputs[:, :, i] for i in range(len(original_val))],
                axis=-1)
        labels = features[:,input_slice,:]
        labels = tf.stack([labels[:, :,len(original_val)]],axis=-1)
        inputs.set_shape([None, input_width, None])
        labels.set_shape([None, input_width, None])
        return inputs, labels

def set_model():
    sequence_input = tf.keras.Input(shape=(input_width,len(original_val)),dtype=tf.float32,name='stock_lstm_input')
    lstm_1 = LSTM(units=32, return_sequences=True ,activation='tanh')(sequence_input)
    dropout_1 = Dropout(0.6)(lstm_1)
    lstm_2 = LSTM(units=32, return_sequences=True ,activation='tanh')(dropout_1)
    dropout_2 = Dropout(0.6)(lstm_2)
    result = Dense(units= 1,kernel_initializer= tf.initializers.zeros(),activation=None)(dropout_2)  
    model = tf.keras.Model(inputs = sequence_input,
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
###############################################################################

_output = pd.DataFrame()
for _year in tqdm(available_years):
    train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < _year ].copy()
    X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(train_df[original_val])
    train_X = np.nan_to_num(X_transformer.transform(train_df[original_val]))
    train_df[original_val] = np.asarray(pd.DataFrame(train_X))
    
    ###########################################################################
    dat = []
    for ticker in tqdm(ticker_sector,leave=False, disable = True):
        temp = train_df[train_df['ticker'] == ticker].copy()    
        temp['date'] = pd.to_datetime(temp['date'])   
        if len(temp) > input_width:
            temp = temp.sort_values('date') 
            data = np.array(temp[original_val+output_val],dtype=np.float32)
            data = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width,
                sequence_stride=1,
                shuffle=False,
                batch_size=1)
            dat.extend(data)
    train_index,test_index = train_test_split(range(0,len(dat)), test_size=0.1,random_state = 1)
    
    _train = tf.data.Dataset.from_tensor_slices( [ tf.reshape(dat[i],[input_width, val_len]) for i in train_index] )
    _validate = tf.data.Dataset.from_tensor_slices( [ tf.reshape(dat[i],[input_width, val_len]) for i in test_index] )
    
    _train = _train.shuffle(len(_train)).batch(1024)
    _validate = _validate.shuffle(len(_validate)).batch(1024)

    _train = _train.map(split_window)
    _validate = _validate.map(split_window)
    
    model.load_weights('saved_models\\initial_weights.h5')
    history = model.fit(_train, epochs=MAX_EPOCHS, validation_data = _validate,verbose=0,
            callbacks=[early_stopping])
    #model.save_weights('saved_models\\' + str(_year) + '_LSTM.h5')
    
    _result= []
    _result_accuracy = []
    for ticker in tqdm(ticker_sector,leave=False, disable = True):
        temp = _dat_[_dat_['ticker'] == ticker].copy()
        temp_current = temp[pd.DatetimeIndex(temp['date']).year == (_year-1) ].copy()    
        temp_current['date'] = pd.to_datetime(temp_current['date'])
        if len(temp_current) >= (input_width-1):
            temp_current = temp_current.iloc[(len(temp_current)-input_width+1):len(temp_current)].copy()
        else:
            temp_current = temp_current
        temp_follow =  temp[pd.DatetimeIndex(temp['date']).year == _year ].copy()
        temp = pd.concat([temp_current,temp_follow], axis = 0)
        if len(temp) > input_width:
            temp['date'] = pd.to_datetime(temp['date'])   
            temp = temp.sort_values('date') 
            test_X = np.nan_to_num(X_transformer.transform(temp[original_val]))
            temp[original_val] = np.asarray(pd.DataFrame(test_X).clip(-4.5,4.5))
            data = np.array(temp[original_val+output_val],dtype=np.float32)
            data = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width,
                sequence_stride=1,
                shuffle=False,
                batch_size=1)
            data = data.map(split_window)
            _pred_y = model.predict(data)
            _pred_y = _pred_y[:,(input_width-1),0]
            _test_y = np.asarray([float(item[1][:,(input_width-1),0]) for item in data])
        
            _result.extend( np.square( (_test_y - _pred_y ) ) ) 
            _result_accuracy.extend(  (_pred_y>=0) == (_test_y>=0) ) 
            temp = temp.iloc[(input_width-1):].copy()
            temp['LSTM_pred'] = _pred_y
            temp = temp[pd.DatetimeIndex(temp['date']).year == _year ].copy()
            _output = pd.concat([_output,temp], axis=0)
            
###############################################################################

_output_2 = _output[pd.DatetimeIndex(_output['date']).year >= 2003]
print('two layer stacked LSTM')
print('MAE')
print( np.mean( np.abs(_output_2['LSTM_pred'] - _output_2['return_t_plus_1']))) 
print('RMSE')
print(np.mean( np.square(_output_2['LSTM_pred'] - _output_2['return_t_plus_1'])))
print('DA')
print(np.mean((_output_2['LSTM_pred']>=0) == (_output_2['return_t_plus_1']>=0)))
print('UDA')
print(sum( np.logical_and((_output_2['LSTM_pred'] >= 0),(_output_2['return_t_plus_1'] >= 0)) ) / sum(_output_2['return_t_plus_1'] >= 0))
print('DDA')
print(sum( np.logical_and((_output_2['LSTM_pred'] < 0),(_output_2['return_t_plus_1'] < 0)) ) / sum(_output_2['return_t_plus_1'] < 0))

#_output = _output[['date','ticker','LSTM_pred']].copy()
#_output.to_csv("saved_output\\LSTM_pred_two_layer_mae_linear_activation_length_4.csv")



