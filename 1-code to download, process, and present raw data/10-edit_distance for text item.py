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

################################################
import json
from tqdm import tqdm
import numpy as np
import multiprocessing
import re
from function.tokenizer import word_clean

###############################################################################
'''regular expression fixing'''

def _word_clean(word):   
    temp = ""
    while temp != word:
        temp = word
        word = re.sub("&"," and ",word)
    
    word = word_clean(word)
    word = " " + ' '.join(word) + " "
    
    temp = ""
    while temp != word:
        temp = word
        word = re.sub("( corporation )|( incorporated )|( bancorporation )|( company )|( companies )|( corp )|( inc )|( plc )|( co )"," inc ",word)
    
    temp = ""    
    while temp != word:
        temp = word
        word = re.sub("( limited )|( ltd )"," inc ",word)
    
    temp = ""
    while temp != word:
        temp = word
        word = re.sub("( holdings )|( holding )|( group )|( groups )"," inc ",word)
    
    temp = ""    
    while temp != word:
        temp = word
        word = re.sub("( global )|( international )|( worldwide )|( world )"," world ",word)
    
    temp = ""
    while temp != word:
        temp = word
        word = re.sub("( association )|( associates )"," association ",word)
    
    temp = ""
    while temp != word:
        temp = word
        word = re.sub("( technologies )|( technology )"," technology ",word)
    
    word = ' '.join(word_clean(word)).title()
    return word    

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
            
'''clean words by regular expression'''
_organizations_keywords = {}
for words in tqdm(organization_keywords):
    _organizations_keywords[words] = {}
    _organizations_keywords[words]['regular'] = _word_clean(words)

###############################################################################
'''get company names'''
company_keywords = json.loads(open('data\\1-ticker_name_list.json').read())

match_list = {}
for ticker in company_keywords:
    match_list[ticker] = []
    match_list[ticker].append(_word_clean(company_keywords[ticker]['wiki_name'][0])) 
    try:
        match_list[ticker].append(_word_clean(company_keywords[ticker]['shortName']))
    except:
        print(ticker)
    try:
        match_list[ticker].append(_word_clean(company_keywords[ticker]['longName']))
    except:
        print(ticker)
    match_list[ticker] = list(set(match_list[ticker]))
        
###############################################################################
             
def lcs(s1, s2):
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
    cs = matrix[-1][-1]
    return len(cs) 

###############################################################################
'''mult core process'''
def f(input_list):
    ticker,match_list,_organizations = input_list[0], input_list[1],input_list[2]
    companies = match_list[ticker]
    Lst = []
    for entity in _organizations:
        company_match = []
        entity_match = []
        for company in companies:   
            regular =  _organizations[entity]['regular']
            length = lcs(company,regular)
            company_match.append(length/len(company))
            entity_match.append(length/len(_organizations[entity]['regular']))   
        Lst.append( {"entity":entity, "regular":regular ,"company_match":company_match,"entity_match":entity_match} )     
    output = sorted(Lst,reverse=True, key = lambda item: np.mean(item['company_match'])+np.mean(item['entity_match']) )[0:20000]       
    return output

###############################################################################

NYT_organization_edit_distance = {}
for ticker in match_list:
    NYT_organization_edit_distance[ticker] = {}

#####################################################################
if __name__ == '__main__':
    num_processor = 16
    pool = multiprocessing.Pool(num_processor)      
    
    def chunks(Lst, n):
       for i in range(0, len(Lst), n):
           yield Lst[i:i + n]             
           
    Lst = []
    for ticker in NYT_organization_edit_distance.keys():
        if NYT_organization_edit_distance[ticker] == {}:
            Lst.append(ticker)
    
    Lst = list(chunks(Lst,num_processor))
        
    for tickers in tqdm(Lst):
        
        input_list = [ [ticker,match_list,_organizations_keywords] for ticker in tickers]
        
        output = pool.map(f,input_list) 
        for ticker,item in zip(tickers,output):
            NYT_organization_edit_distance[ticker] = item

    with open('data\\10-NYT_keyword_edit_distance.json', 'w') as fp:
        json.dump(NYT_organization_edit_distance, fp)  

