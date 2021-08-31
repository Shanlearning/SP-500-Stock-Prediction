import os
from pathlib import Path
project_dir = Path.cwd().parent
os.chdir(project_dir)
#os.chdir('change to the mother working directory')

os.chdir('C:\\Users\\zhong\\Dropbox\\github\\stock_project_for_sbumission')
###############################################################################
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
###############################################################################
ticker_sector = json.loads(open('2-cleaned_data\\ticker_sector_information.json').read())
_sector = []
for ticker in ticker_sector:
    _sector.append(ticker_sector[ticker]['sector'])
_sector = list(set(_sector))

###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
###############################################################################
'''correlation plot among variables'''

corr = _dat_[['return_t','cci','macdh','rsi_14','kdjk' ,'wr_14','cmf','atr_percent','sentiment','PeRatio', 'PsRatio', 'PbRatio']]
f = plt.figure(figsize=(19, 15))
plt.matshow(corr.corr(), fignum=f.number)
plt.xticks(range(corr.select_dtypes(['number']).shape[1]), corr.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(corr.select_dtypes(['number']).shape[1]), corr.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

###############################################################################
''' plots for processed variables against the return at t+1'''

_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)

vals = ['return_t','sentiment','PeRatio', 'PsRatio', 'PbRatio','cci','macdh','rsi_14','kdjk' ,'wr_14','atr_percent','cmf']

_dat_2 = _dat_.copy()
X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(_dat_2[vals])
_dat_2 = pd.DataFrame(np.nan_to_num(X_transformer.transform(_dat_2[vals])))
vals = ['return_t','sentiment','PE', 'PS', 'PB','CCI','MACDH','RSI','KDJ' ,'WR','ATR','CMF']
_dat_2.columns = vals

plt.figure(figsize = (8,6))
gs1 = gridspec.GridSpec(3, 4)

ax0 = plt.subplot(gs1[0])
ax0.set_xlim(-3,3)
ax0.set_ylim(-0.1,0.1)

for i in range(0,len(vals)):
    ax1 = plt.subplot(gs1[i], sharex = ax0,sharey = ax0)
    val = vals[i]
    x = _dat_2[val]; y = _dat_['return_t_plus_1']
    ax1.scatter(x,y,s = 0.1,alpha = 0.03)    
    m, b = np.polyfit(x, y, 1)
    x = np.linspace(-4,4,100)
    ax1.plot(x, m*x + b,color="red", linewidth=0.8,linestyle='-.')  
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.text(-2.95,0.065,val)
plt.subplots_adjust(wspace=0, hspace=0)    
plt.savefig('val_to_return.png', dpi=400,bbox_inches='tight') 

###############################################################################
''' two plots for the sentiment'''

_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)

val = "sentiment"

plt.figure(figsize = (6,4))
gs1 = gridspec.GridSpec(4, 5)

ax0 = plt.subplot(gs1[0])
ax0.set_xlim(-3.5,3.5)
ax0.set_ylim(-0.1,0.1)

start_years = list(range(2000,2020,1))
M_val = []
b_val = []
for i in range(0,len(start_years)):
    year = start_years[i]
    temp = _dat_[pd.DatetimeIndex(_dat_['date']).year>=year].copy()
    temp = temp[pd.DatetimeIndex(temp['date']).year<=(year)].copy()
    temp = temp[~temp[val].isna()].copy()
    X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(temp[[val]])
    x = X_transformer.transform(np.nan_to_num(temp[[val]])).ravel() ; y = temp['return_t_plus_1']
    ax1 = plt.subplot(gs1[i], sharex = ax0,sharey = ax0)
    ax1.scatter(x,y,s = 0.1,alpha = 0.3)    
    m, b = np.polyfit(x, y, 1)
    x = np.linspace(-6,6,100)
    ax1.plot(x, m*x + b,color="red", linewidth=0.8,linestyle='-.')  
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.text(-2.95,0.065,str(year)+"-"+str(year+1))
    M_val.append(m)
    b_val.append(b)
plt.subplots_adjust(wspace=0, hspace=0)    
plt.savefig('sentiment_to_return.png', dpi=800)    


plt.figure(figsize = (24,4))
ax = plt.figure().gca()
ax.plot(start_years,M_val,color="blue")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.subplots_adjust(wspace=0, hspace=0)    
plt.savefig('parameter plot.png', dpi=800)  

###############################################################################
_dat_ = pd.read_csv("2-cleaned_data\\dat_518_companies.csv",index_col = 0)
###############################################################################
#_dat_ = _dat_[pd.DatetimeIndex(_dat_['date']).year>=2007].copy()
vals = ['PbRatio']
for val in vals:
    _dat_ = _dat_[~_dat_[val].isna()].copy()
    X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(_dat_[[val]])
    x = X_transformer.transform(np.nan_to_num(_dat_[[val]])).ravel()
    _dat_[val] = x

plt.figure(figsize = (10,10))
gs1 = gridspec.GridSpec(3, 4)

ax0 = plt.subplot(gs1[0])
ax0.set_xlim(-3.5,3.5)
ax0.set_ylim(-0.1,0.1)
i = 0
M_val = {}
for sector in _sector:
    ax1 = plt.subplot(gs1[i], sharex = ax0,sharey = ax0)
    dat_sector= _dat_[_dat_['sector'] == sector].copy()
    M_val[sector] = []
    for ticker in ticker_sector:
        temp = dat_sector[dat_sector['ticker'] == ticker].copy()
        if len(temp)>2 :
            cols = ['red']
            col = 0
            for val in vals:
                x = temp[val]
                y = temp['return_t_plus_1']
                m, b = np.polyfit(x, y, 1)
                M_val[sector].append(m)
                x = np.linspace(-4,4,100)
                ax1.plot(x, m*x + b,color=cols[col], linewidth=0.8,linestyle='-.',alpha = 0.7)  
                col = col + 1
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.text(-2.95,0.065,str(sector))
    i = i+1
plt.subplots_adjust(wspace=0, hspace=0)  
plt.savefig('val_to_return.png', dpi=800)

for sector in _sector:
    print(sector)
    print(np.mean([item <0 for item in M_val[sector]]))