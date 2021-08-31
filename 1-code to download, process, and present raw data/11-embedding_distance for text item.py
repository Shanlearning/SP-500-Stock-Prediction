# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:16:57 2020

@author: zhong
"""

################################################
import sys, os
from pathlib import Path
project_dir = Path.cwd()
os.chdir(project_dir)
sys.path.append(os.getcwd())
#os.chdir('change to the mother working directory')

###############################################################################
import json
from tqdm import tqdm
from function.tokenizer import word_clean,load_vocab,word_segment
from tensorflow.keras.losses import cosine_similarity

# pip install --upgrade tensorflow-hub
import tensorflow_hub as hub
import numpy as np
###############################################################################
''' download from https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1'''
vocab = load_vocab("model\\nnlm_dim_128\\assets\\tokens.txt")
###############################################################################
'''load data by month, get organization keywords names'''
organization_keywords = []

for year in tqdm(range(2000,2020)):
    for month in range(1,13):
        data = json.loads(open('data\\8-NYT_data\\' +str(year) + "_" + str(month) + '.json').read())
        for article in data['response']['docs']:           
            for keyword in article['keywords']:
                if keyword['name'] in ["organizations"]:        
                    organization_keywords.append(keyword['value'])
                    
organization_keywords = list(set(organization_keywords))            

_organization_keywords = {}
for words in tqdm(organization_keywords):
    _organization_keywords[words] = {}

'''make sure every word is in the vocab'''
for words in tqdm(_organization_keywords):
    tokens = word_clean(words,do_lower_case = False)
    temp = []
    for token in tokens:
        if token in vocab:
            temp.append(token)
        else:
            temp_token = " ".join(word_clean(token,do_lower_case = True)).title()
            if temp_token in vocab:
                temp.append(temp_token)
            else:
                temp_word_1 = " ".join(word_clean(token,do_lower_case = True))
                if  temp_word_1 in vocab:
                    temp.append(temp_word_1)
                else:
                    case_1 = word_segment( token, vocab = vocab )
                    case_2 = word_segment( temp_token , vocab = vocab )
                    case_3 = word_segment( temp_word_1 , vocab = vocab )
                    min_len = min([len(item) for item in [case_1,case_2,case_3]])    
                    candidate = [list(item) for item in [case_1,case_2,case_3] if len(item) == min_len]
                    temp.extend( sorted(candidate,key = lambda words: np.std([len(word) for word in words]))[0] )
    _organization_keywords[words]['embed'] = temp

###############################################################################
'''get company names'''
company_keywords = json.loads(open('data\\1-ticker_name_list.json').read())

match_list = {}
for ticker in company_keywords:
    match_list[ticker] = []
    match_list[ticker].append(" ".join( word_clean( company_keywords[ticker]['wiki_name'][0] ) ) ) 
    try:
        match_list[ticker].append(" ".join( word_clean( company_keywords[ticker]['shortName'] ) ) )
    except:
        print(ticker)
    try:
        match_list[ticker].append(" ".join( word_clean( company_keywords[ticker]['longName'] ) ) )
    except:
        print(ticker)
    match_list[ticker] = list(set(match_list[ticker]))
        
###############################################################################

embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1")

###############################################################################

NYT_organization_edit_distance = json.loads(open('data\\10-NYT_keyword_edit_distance.json').read())

def f(ticker):
    companies = []
    for company_embed in match_list[ticker]:   
        companies.append( embed([company_embed]) )
    output = []
    _ticker_organizations = NYT_organization_edit_distance[ticker]
    for i in range(0, len(_ticker_organizations) ):
        entity = _ticker_organizations[i]['entity']
        Embed_entity = " ".join( _organization_keywords[entity]['embed'] )
        temp = {}
        embeddings = embed( [Embed_entity] )
        embedding_match = []
        for company in companies:
            embedding_match.append(cosine_similarity(company[0],embeddings[0],axis=-1).numpy())
        temp['entity'] = entity
        temp['average_embedding_score'] = np.mean(embedding_match)
        output.append(temp)
        
    output_2 = sorted(output,reverse=True, key = lambda item: item['average_embedding_score'] )
    
    for i in range(0,len(output_2)):
        output_2[i]['average_embedding_score'] = float(output_2[i]['average_embedding_score']) 
    return output_2[0:50]

###############################################################################

filtered_list = json.loads(open('data\\5-filtered_ticker_list.json').read())

Lst = []
for ticker in filtered_list.keys():
    Lst.append(ticker)
    
NYT_keyword_embedding_distance = {}
for i in tqdm(range(len(Lst))):
    ticker = Lst[i]
    output = f(ticker)
    NYT_keyword_embedding_distance[Lst[i]] = output
    
    if (i in [0,50,100,150,200,250,300,350,400,450,len(Lst)-1]):
        with open('data\\11-NYT_keyword_embedding_distance.json', 'w') as fp:
            json.dump(NYT_keyword_embedding_distance, fp)  
















