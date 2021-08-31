# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:01:36 2020

@author: zhong
"""

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

################################################
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import pandas as pd

from datetime import datetime
from collections import Counter
###############################################################################
'count number of all articles'

def get_type_of_material(data):
    type_of_material = []
    for i in range(0,len(data['response']['docs'])):
        try:
            type_of_material.append(data['response']['docs'][i]['type_of_material'])
        except:
            type_of_material.append("News")
    return type_of_material

output = {}

for year in tqdm(range(2000,2020)):
    cnt = Counter()
    for month in range(1,13):
        data = json.loads(open('data\\8-NYT_data\\' +str(year) + "_" + str(month) + '.json').read())
        data = get_type_of_material(data)
        for item in data:
            cnt[item] += 1    
    output[datetime(year, month, 1).strftime("%Y")] = dict(cnt)
    
###############################################################################
names = []
for date in output.keys():
    names.extend(list(output[date].keys()))
names = np.unique(names)

temp = {}
for name in names:
    temp[name] = {}
    for date in list(output.keys()):
        try:
            temp[name][date] = output[date][name]
        except:
            temp[name][date] = 0

cnt = Counter()
for name in list(temp.keys())[:-1]:
    cnt[name] = sum( temp[name].values() )
cnt.most_common(50)


cnt = Counter()
for name in list(temp.keys())[:-1]:
    if name in ['News', 'News Analysis','Newsletter','Brief','briefing','Caption','First Chapter']:
        cnt['News'] = cnt['News'] + sum(temp[name].values())
    if name in  ['An Appraisal','Review', 'Review; List','Review; Series', 'Review; Text' , 'Text; Review' , 'List; Review', 'Op-Ed', 'Op-Ed; Caption', 'Op-Ed; Series', 'Op-Ed; Text','Caption; Op-Ed']:
        cnt['Review; Opinion'] = cnt['Review; Opinion'] + sum(temp[name].values())
    if name in ['Audio Podcast','Slideshow','Video', 'Web Log', 'Live Blog Post','Interactive Feature', 'Interactive Graphic','Sidebar']:
        cnt['Video; Blog; Slideshow'] = cnt['Video; Blog; Slideshow'] + sum(temp[name].values())
    if name in  ['Biography','Paid Death Notice','Biography; Chronology','Biography; Obituary','Series; Biography', 'Biography; Series','Obituary', 'Obituary (Obit)', 'Obituary; Biography']:
        cnt['Obituary; Biography'] = cnt['Obituary; Biography'] + sum(temp[name].values())
    if name in ['Letter']:
        cnt['Letter'] = cnt['Letter'] + sum(temp[name].values())
    if name in [ 'Correction',"Correction; Editors' Note", "Correction; Editors' Note", 'Correction; Series' ]:
        cnt['Correction'] = cnt['Correction'] + sum(temp[name].values())
    if name in ['An Analysis', 'An Analysis; Economic Analysis',
       'An Analysis; Military Analysis', 'An Analysis; News Analysis',
       'An Analysis; News Analysis; Chronology','Economic Analysis','Military Analysis',
       'Special Report','Statistics','Summary','Text','List','Series','Series; Text']:
        cnt['Analysis; Report; Summary'] = cnt['Analysis; Report; Summary'] + sum(temp[name].values())
    if name in ['Postscript','Premium','Profile','Quote','Results Listing','Schedule','Transcript','Glossary','others', 'recipe','Addendum','QandA','Question',
 'Editorial', 'Editorial; List', 'Editorial; Series', 'Editors Note', "Editors' Note",'Series; Editorial','Caption; Editorial','Chronology','Chronology; An Analysis; News Analysis', 
 'Series; Chronology' ,'Chronology; Series','Chronology; Special Report','List; Chronology','Special Report; Chronology',
 'Interview', 'Interview; Review', 'Interview; Series', 'Interview; Text', 'Text; Interview', 'Series; Interview']:
         cnt['Others'] = cnt['Others'] + sum(temp[name].values())

print(cnt.most_common(50))
###############################################################################

numbers = []
labels = []
for name in [
            'News',
            'Review; Opinion',
            'Video; Blog; Slideshow',
            'Obituary; Biography',
            'Letter',
            'Correction',
            'Analysis; Report; Summary',
            'Others']:
    numbers.append( cnt[name] )
    labels.append(name)

percents = ["{0:.1%}".format( number / sum(numbers) ) for number in numbers]

dates = []
hits = []
for date in output.keys():
    dates.append(date)
    hits.append( sum( output[date].values() ) )
    
###############################################################################

temp = pd.DataFrame(temp)
temp_plot = pd.DataFrame()

temp_plot['News'] = temp[['News', 'News Analysis','Newsletter','Brief','briefing','Caption','First Chapter']].T.sum()
temp_plot['Review; Opinion'] = temp[['An Appraisal','Review', 'Review; List','Review; Series', 'Review; Text' , 'Text; Review' , 'List; Review', 'Op-Ed', 'Op-Ed; Caption', 'Op-Ed; Series', 'Op-Ed; Text','Caption; Op-Ed']].T.sum()
temp_plot['Video; Blog; Slideshow'] = temp[['Audio Podcast','Slideshow','Video', 'Web Log','Interactive Feature', 'Interactive Graphic','Sidebar']].T.sum() #'Live Blog Post',
temp_plot['Obituary; Biography'] = temp[['Biography','Paid Death Notice','Biography; Chronology','Biography; Obituary','Series; Biography', 'Biography; Series','Obituary', 'Obituary (Obit)', 'Obituary; Biography']].T.sum()
temp_plot['Letter'] = temp[['Letter']].T.sum()
temp_plot['Correction'] = temp[['Correction',"Correction; Editors' Note", "Correction; Editors' Note", 'Correction; Series']].T.sum()
temp_plot['Analysis; Report; Summary'] = temp[['An Analysis', 'An Analysis; Economic Analysis',
       'An Analysis; Military Analysis', 'An Analysis; News Analysis',
       'An Analysis; News Analysis; Chronology','Economic Analysis','Military Analysis',
       'Special Report','Statistics','Summary','Text','List','Series','Series; Text']].T.sum()
temp_plot['Others'] = temp[['Postscript','Premium','Profile','Quote','Results Listing','Schedule','Transcript','Glossary', 'recipe','Addendum','QandA','Question',
 'Editorial', 'Editorial; List', 'Editorial; Series', 'Editors Note', "Editors' Note",'Series; Editorial','Caption; Editorial','Chronology','Chronology; An Analysis; News Analysis', 
 'Series; Chronology' ,'Chronology; Series','Chronology; Special Report','List; Chronology','Special Report; Chronology',
 'Interview', 'Interview; Review', 'Interview; Series', 'Interview; Text', 'Text; Interview', 'Series; Interview']].T.sum()

temp_plot = temp_plot[1:21].to_dict()

dates = dates[1:21]

fig1, ax = plt.subplots(nrows=1, ncols=2,figsize=(16.50764,7))
ax[0].pie(numbers,startangle=90,shadow=True,labels = labels,textprops={'fontsize': 6.5}, autopct='%.0f%%')
ax[0].legend([label + ": " + percent  for label,percent in zip(labels, percents)],loc="upper left",framealpha = 0.93,prop={'size': 9})
ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax[1].yaxis.grid(linestyle='-.',linewidth = 0.4)
ax[1].bar(dates,list(temp_plot[labels[0]].values()),alpha = 0.93)
stack_sum = list(temp_plot[labels[0]].values())
for name in labels[1:]:
    ax[1].bar(dates, temp_plot[name].values(), bottom = stack_sum, label = name, alpha = 0.7)
    stack_sum = np.sum ( [stack_sum , list(temp_plot[name].values())] , axis = 0)
ax[1].tick_params(axis='x', labelrotation=45)

ax[0].set_title('Summary for NYT article types from 2000-01-01 to 2019-12-31',color = "darkslategrey",ha='center', y = 1.05)
ax[1].set_title('Yearly NYT article count from 2000-01-01 to 2019-12-31',color = "darkslategrey",ha='center', y = 1.05)
plt.savefig('NYT_summary.png', dpi=300)
plt.show()