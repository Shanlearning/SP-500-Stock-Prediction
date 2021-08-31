# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:37:59 2020

@author: zhong
"""

###############################################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

from tqdm import tqdm
import json
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from function.bert_setting import predict_project_special

import pandas as pd

###############################################################################
'''load the saved model'''
model = BertForSequenceClassification.from_pretrained('./model/output_model/stock', cache_dir=None, num_labels=3)

###############################################################################

dat = json.loads(open('data\\12-all_500_companies_news.json').read())
dat = pd.DataFrame(dat)

output = []
for i in tqdm(range(0,len(dat))):    
    article = dat.iloc[i]
    text = article['main_headline']
    text = text + " " + article['print_headline']
    text = text + " " +  article['snippet']
    text = text + " " +  article['lead_paragraph']
    result = predict_project_special(text,model)
    temp = dat.iloc[i].copy()
    temp['prediction'] = result.iloc[0]['prediction']
    temp['logit'] = list(result.iloc[0]['logit'])
    temp = temp.to_dict()
    temp['logit'] = [float(number) for number in temp['logit']]
    temp['rank'] = int(temp['rank'])
    output.append(temp)

with open('data\\14-cleaned_all_500_company_news_sentiment.json', 'w') as fp:
    json.dump(output, fp)
    
###############################################################################   
    
dat = json.loads(open('data\\12-sp_500_index_news.json').read())
dat = pd.DataFrame(dat)

output = []
for i in tqdm(range(0,len(dat))):    
    article = dat.iloc[i]
    text = article['main_headline']
    text = text + " " + article['print_headline']
    text = text + " " +  article['snippet']
    text = text + " " +  article['lead_paragraph']
    result = predict_project_special(text,model)
    temp = dat.iloc[i].copy()
    temp['prediction'] = result.iloc[0]['prediction']
    temp['logit'] = list(result.iloc[0]['logit'])
    temp = temp.to_dict()
    temp['logit'] = [float(number) for number in temp['logit']]
    temp['rank'] = int(temp['rank'])
    output.append(temp)

with open('data\\14-cleaned_sp_500_index_sentiment.json', 'w') as fp:
    json.dump(output, fp)
    