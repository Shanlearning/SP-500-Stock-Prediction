

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

##################################
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
##################################

_stock_AV = json.loads(open('data\\_cleaned_stock_data\\stock_data_AV.json').read())
_stock_YF = json.loads(open('data\\_cleaned_stock_data\\stock_data_YF.json').read())

###############################################################################
'''problematic data'''
del _stock_AV['LUMN']
del _stock_AV['DNR']
del _stock_AV['LM']
del _stock_AV['NE']

del _stock_YF['LUMN']
del _stock_YF['DNR']
del _stock_YF['LM']
del _stock_YF['NE']

###############################################################################
'''check length of dates matching'''

AV_length = []
for ticker in tqdm( _stock_AV.keys() ):
    AV_date = [date for date in list( _stock_AV[ticker].keys() )  if 
               datetime.strptime(date,'%Y-%m-%d') >=  datetime(1999, 11, 1, 0, 0) and  datetime.strptime(date,'%Y-%m-%d') <=  datetime(2019, 12, 31, 0, 0)]
    AV_length.append(len(AV_date))
    
YF_length = []
for ticker in tqdm( _stock_AV.keys() ):
    YF_date = [date for date in list( _stock_YF[ticker].keys() )  if 
               datetime.strptime(date,'%Y-%m-%d') >=  datetime(1999, 11, 1, 0, 0) and  datetime.strptime(date,'%Y-%m-%d') <=  datetime(2019, 12, 31, 0, 0)]
    YF_length.append(len(YF_date))

temp = np.absolute(np.asarray(AV_length)-np.asarray(YF_length))
sum(temp==0)

print(sum(temp==1))
print(sum(temp>1))
ticker_list = np.asarray( list(_stock_AV.keys()) )
ticker_list[temp>1]

#list(_stock_AV.keys())[np.where([temp>1])[1][0]]

###############################################################################
'''accuracy within matched dates'''
'''YF data have Open, High, Low, Close adjusted by split, while AV data do not'''

dat = {} 
for ticker in tqdm(_stock_AV.keys()):
    AV = pd.DataFrame.from_dict(_stock_AV[ticker]).T
    AV.index = pd.to_datetime(AV.index)
    AV = AV.loc[ AV.index <= datetime(2020, 9, 30, 0, 0)]
    AV = AV.loc[ AV.index >= datetime(1999, 11, 1, 0, 0)]
    
    YF = pd.DataFrame.from_dict(_stock_YF[ticker]).T
    YF.index = pd.to_datetime(YF.index)
    YF = YF.loc[ YF.index <= datetime(2020, 9, 30, 0, 0)]
    YF = YF.loc[ YF.index >= datetime(1999, 11, 1, 0, 0)]
    
    merge = pd.merge(AV,YF,left_index=True,right_index=True,how = "inner")
    merge = merge.astype(float)
                      
    split = np.array(merge['Stock Splits'])
    split[split==0] = 1
    split_efficient_YF = np.cumprod(1/split)
    split_efficient_YF = np.concatenate([np.array([1]),split_efficient_YF[:-1]])
    
    '''adjust price by split'''
    merge['Open'] = merge['Open'] /  split_efficient_YF
    merge['High'] = merge['High'] /  split_efficient_YF
    merge['Low'] = merge['Low'] /  split_efficient_YF
    merge['Close'] = merge['Close'] /  split_efficient_YF
    
    dat[ticker] = merge
    
#################################
'''reliability of dividends and splits data'''

bad_report = {}
for ticker in tqdm(_stock_AV.keys()):
    bad_report[ticker] = {}
    merge = dat[ticker]
    '''date of YF splits'''
    temp_YF = set(merge.index[merge['Stock Splits']!=0]) 
    '''date of AV splits'''
    temp_AV = set(merge.index[merge['8. split coefficient']!=1])
    bad_report[ticker]["Total stock splits dates"] =  max(len(temp_YF),len(temp_AV))
    
    '''count for the date matched part'''
    test = merge.loc[np.logical_and(merge['Stock Splits']!=0, merge['8. split coefficient']!=1)]    
    if len(test) > 0:
        bad_report[ticker]["Matched splits dates count"] = len(test)
        bad_report[ticker]["Splits size exactly match"] = sum(test['Stock Splits'] == test['8. split coefficient'])
        bad_report[ticker]["Splits size within 1% error"] = sum(np.logical_and(test['8. split coefficient']*1.01 >= test['Stock Splits'],
                                                                          test['8. split coefficient']*0.99 <= test['Stock Splits']))
    else:
        bad_report[ticker]["Matched splits dates count"] = 0
        bad_report[ticker]["Splits size exactly match"] = 0
        bad_report[ticker]["Splits size within 1% error"] = 0
    
    '''date of YF dividents'''
    temp_YF = set(merge.index[merge['Dividends']!=0]) 
    '''date of AV dividents'''
    temp_AV = set(merge.index[merge['7. dividend amount']!=0])
    bad_report[ticker]["Total dividends dates"] =  max(len(temp_YF),len(temp_AV))
    '''count for the date matched part'''
    test = merge.loc[np.logical_and(merge['Dividends']!=0, merge['7. dividend amount']!=0)]  
    if len(test) > 0:
        bad_report[ticker]["Matched dividend dates count"] = len(test)
        bad_report[ticker]["Dividends amount exactly match"] = sum(test['Dividends'] == test['7. dividend amount'])
        bad_report[ticker]["Dividends amount within 1% error"] = sum(np.logical_and(test['7. dividend amount']*1.01 >= test['Dividends'],
                                                                          test['7. dividend amount']*0.99 <= test['Dividends']))
    else:
        bad_report[ticker]["Matched dividend dates count"] = 0
        bad_report[ticker]["Dividends amount exactly match"] = 0
        bad_report[ticker]["Dividends amount within 1% error"] = 0
    
#######################
'''count together for the splits and dividends'''

bad_report = pd.DataFrame(bad_report).T
bad_report = bad_report.sum()

print(bad_report)
    
###############################################################################
''' accuracy report for the stocks'''
bad_report = {}
for ticker in _stock_AV.keys():
    bad_report[ticker] = {}
    merge = dat[ticker]    
    bad_report[ticker]['len_ticker'] = len(merge)
       
    ################################
    bad_report[ticker]['Open'] ={}       
    bad_report[ticker]['Open']['Percent within 1% error'] =np.mean( np.logical_and(merge['1. open']*1.01 >= merge['Open'],
                                                                            merge['1. open']*0.99 <= merge['Open']) )   
    bad_report[ticker]['Open']['99% quantile error'] = np.quantile( np.absolute( merge['1. open']-merge['Open'] ) / merge['Open'] , 0.99 ) 
    ################################
    bad_report[ticker]['High'] = {}
    bad_report[ticker]['High']['Percent within 1% error'] =np.mean( np.logical_and(merge['2. high']*1.01 >= merge['High'],
                                                                            merge['2. high']*0.99 <= merge['High']) )
    bad_report[ticker]['High']['99% quantile error'] = np.quantile( np.absolute( merge['2. high']-merge['High'] ) / merge['High'] , 0.99 ) 
    ################################   
    bad_report[ticker]['Low'] ={}
    bad_report[ticker]['Low']['Percent within 1% error'] =np.mean( np.logical_and(merge['3. low']*1.01 >= merge['Low'],
                                                                           merge['3. low']*0.99 <= merge['Low']) )
    bad_report[ticker]['Low']['99% quantile error'] = np.quantile( np.absolute( merge['3. low']-merge['Low'] ) / merge['Low'] , 0.99 ) 
    ################################
    bad_report[ticker]['Close'] ={}
    bad_report[ticker]['Close']['Percent within 1% error'] =np.mean( np.logical_and(merge['4. close']*1.01 >= merge['Close'],
                                                                           merge['4. close']*0.99 <= merge['Close']) )
    bad_report[ticker]['Close']['99% quantile error'] = np.quantile( np.absolute( merge['4. close']-merge['Close'] ) / merge['Close'] , 0.99 ) 
    ################################
    bad_report[ticker]['Volume'] = {}
    bad_report[ticker]['Volume']['Percent within 1% error'] = np.mean( np.logical_and(merge['6. volume']*1.01 >= merge['Volume'],
                                                                               merge['6. volume']*0.99 <= merge['Volume']) )
    bad_report[ticker]['Volume']['99% quantile error'] = np.quantile( np.absolute( merge['6. volume']-merge['Volume'] ) / merge['Volume'] , 0.99 ) 
    ################################
    bad_report[ticker]['Adj Close'] = {}
    bad_report[ticker]['Adj Close']['Percent within 1% error'] = np.mean( np.logical_and(merge['5. adjusted close']*1.01 >= merge['Adj Close'],
                                                                                  merge['5. adjusted close']*0.99 <= merge['Adj Close']) )
    bad_report[ticker]['Adj Close']['99% quantile error'] = np.quantile( np.absolute( merge['5. adjusted close']-merge['Adj Close'] ) / merge['Adj Close'] , 0.99 ) 
    
###############################################################################

for val in ['Open','High','Low','Close','Volume','Adj Close']:
    print(val)
    report = []
    sums = []
    for ticker in bad_report.keys():
        report.append(bad_report[ticker][val]['Percent within 1% error'] * bad_report[ticker]['len_ticker'])
        sums.append(bad_report[ticker]['len_ticker'])
    print('Percentage of data points within 1% error: ' +str(sum(report)/sum(sums)) )
    
    report = 0
    for ticker in bad_report.keys():
        if bad_report[ticker][val]['99% quantile error'] < 0.05:
            report = report +1
    print('Percentage of stocks have 99% quantile error less than 5%: ' + str(report/len(bad_report.keys())) )
    

