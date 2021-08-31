# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:15:25 2020

@author: zhong
"""

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

import urllib.request, json 
import time
import os

'''register for a free api key from NYT website: https://developer.nytimes.com/apis'''

Path('data\\8-NYT_data\\').mkdir(parents=True, exist_ok=True)

for year in range(2000,2020):    
    for month in range(1,13):
        if not os.path.isfile('data\\8-NYT_data\\' +str(year) + "_" + str(month) + '.json'):   
            with urllib.request.urlopen("https://api.nytimes.com/svc/archive/v1/" + str(year) + "/" + str(month) + ".json?api-key=your api key") as url:
                data = json.loads(url.read().decode())
            with open('data\\8-NYT_data\\' +str(year) + "_" + str(month) + '.json', 'w') as f:
                json.dump(data, f)
            time.sleep(120)
        time.sleep(6)
        


