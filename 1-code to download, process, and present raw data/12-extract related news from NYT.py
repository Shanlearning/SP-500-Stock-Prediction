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

###############################################################################


from tqdm import tqdm
import json

###############################################################################
'''these results were mannually checked from the embedding distance outputs'''

match_list = json.loads(open('data\\11-NYT_ticker_company_name.json').read())

###############################################################################
'''load NYT data'''
Lst = []
for year in tqdm(range(1999,2020)):
    for month in range(1,13):
        data = json.loads(open('data\\NYT _data\\' +str(year) + "_" + str(month) + '.json').read())
        for article in data['response']['docs']:           
            temp= {}
            temp['keywords'] = article['keywords']
            temp['headline'] = article['headline']
            if 'lead_paragraph'in article:
                temp['lead_paragraph'] = article['lead_paragraph']
            else:
                temp['lead_paragraph'] = ""
            temp['snippet'] = article['snippet']
            temp['pub_date'] = article['pub_date']
            if 'type_of_material' in article:
                temp['type_of_material'] = article['type_of_material']
            else:
                 temp['type_of_material'] = "not exist"
            temp['document_type'] = article['document_type']
            Lst.append( temp )

###############################################################################
'''extract related articles'''
output = []
for ticker in tqdm(match_list):
    for item in Lst:
        article = item
        for keyword in item['keywords']:
            if (keyword['value'] in match_list[ticker]['organization'])&(keyword['name'] == "organizations"):   
                temp = dict()
                temp['main_headline'] = article['headline']['main'] 
                
                if article['headline']['print_headline'] is not None:
                    if article['headline']['print_headline'] != article['headline']['print_headline'] :        
                        temp['print_headline'] = article['headline']['print_headline'] 
                    else:
                        temp['print_headline'] = ""        
                else:
                   temp['print_headline'] = ""
                temp['snippet'] = article['snippet']
                
                if 'lead_paragraph' in article:
                    temp['lead_paragraph'] = article['lead_paragraph']
                else:
                    temp['lead_paragraph'] = ""
                    
                temp['pub_date'] = article['pub_date']
                
                if 'type_of_material' in article:
                    temp['type_of_material'] = article['type_of_material']
                else:
                    temp['type_of_material'] = "News"
                temp['document_type'] = article['document_type']
                temp['ticker'] = ticker
                temp['rank'] = keyword['rank']
                output.append(temp)
          
with open('data\\12-all_500_companies_news.json', 'w') as fp:
    json.dump(output, fp)   

###############################################################################
'''extract articles for the index news'''
sp_news = []
for item in tqdm(Lst):
    for keyword in item['keywords']:
        keyword
        if keyword['value'] in ["Standard & Poor's 500-Stock Index","STANDARD & POOR'S STOCK INDEX"]:        
            sp_news.append(item)

output = []
for article in tqdm(sp_news):         
    temp = dict()
    temp['main_headline'] = article['headline']['main'] 
    
    if article['headline']['print_headline'] is not None:
        if article['headline']['print_headline'] != article['headline']['print_headline'] :        
            temp['print_headline'] = article['headline']['print_headline'] 
        else:
            temp['print_headline'] = ""        
    else:
       temp['print_headline'] = ""
    temp['snippet'] = article['snippet']
    
    if 'lead_paragraph' in article:
        temp['lead_paragraph'] = article['lead_paragraph']
    else:
        temp['lead_paragraph'] = ""
        
    temp['pub_date'] = article['pub_date']
    
    if 'type_of_material' in article:
        temp['type_of_material'] = article['type_of_material']
    else:
        temp['type_of_material'] = "News"
    temp['document_type'] = article['document_type']
    
    for keyword in article['keywords']:
        if keyword['value'] in ["Standard & Poor's 500-Stock Index","STANDARD & POOR'S STOCK INDEX"]:
            temp['rank'] = keyword['rank']
            output.append(temp)
            
with open('data\\12-sp_500_index_news.json', 'w') as fp:
    json.dump(output, fp)        