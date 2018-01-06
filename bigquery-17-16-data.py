#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:18:27 2018

@author: emg
"""

import scipy as sp
import spacy
import pandas as pd
import pickle
import gzip
from tqdm import tqdm
from google.cloud import bigquery
from logging import getLogger
from concurrent.futures import TimeoutError
import sys
import os
from pathlib import Path
import pandas as pd

PROJECT = 'reddit-network-184710'
CREDENTIALS = 'reddit-network-774059619c28.json'

CACHE = 'cache'

def client():
    return bigquery.Client.from_service_account_json(CREDENTIALS, project=PROJECT)

config = bigquery.QueryJobConfig()
config.query_parameters = (bigquery.ScalarQueryParameter('size', 'INT64', 10),)

config = config or bigquery.QueryJobConfig()
config.use_legacy_sql = False
config.maximum_bytes_billed = int(5e9)

def run_job(query):
    print('Submitting query')
    j = client().query(query=query, job_config=config)
    with tqdm() as pbar:
        while True:
            try:
                j.result(timeout=1)
            except TimeoutError:                
                pbar.update(1)
            else:
                break
    return j

def unpack_results(j):
    print('Unpacking results')
    
    total = j.query_results().total_rows
    
    iterator = j.result()
    rows = []
    for row in tqdm(iterator, total=total):
        rows.append(row.values())
    
    columns = [c.name for c in iterator.schema]
    df = pd.DataFrame(rows, None, columns)
    
    return df

# GET MONTHLY COMMENTS FOR BOTH EGO SUBREDDITS
def ego_comments():
    query = """SELECT subreddit, author, created_utc
                FROM `fh-bigquery.reddit_comments.2017_06`
                WHERE subreddit IN ('changemyview', 'The_Donald')"""
    j = run_job(query)
    df = unpack_results(j)
    
    return df

def save_ego_comments(df):
    df.to_csv('cmv-td-17-06-comments.csv')
    df.to_pickle('cmv-td-17-06-comments.pkl')

def load_ego_comments():
    return pd.read_pickle('cmv-td-17-06-comments.pkl')

def split_list(List, parts):
    size = len(List)//parts
    sublists = []
    start, end = 0, size
    for i in range(parts-1):
        sublists.append(List[start:end])
        start += size
        end += size
    sublists.append(List[start:])
    
    n = 0
    for sub in sublists:
        n += len(sub)
    assert n == len(List), "sublists and original lists don't have same number of items"
    
    return sublists

def cocomments_queries(df):
    '''full list of authors exceeds query character length so divide into sublists'''
    authors = list(set(df['author']))

    sublists = split_list(authors, 4)

    queries = []
    print('Building queries...')
    for sub in sublists:
        s = ''
        for name in sub:
            s = s + "'" + name + "' ,"
        
        s = s[:-2]
        
        query = """SELECT subreddit, author, created_utc
                    FROM `fh-bigquery.reddit_comments.2017_06`
                    WHERE author IN ({})""".format(s)
        queries.append(query)

    return queries

def all_cocomments():
    queries = cocomments_queries()
    dfs = []
    for query in queries:
        j = run_job(query)
        print()
        df = unpack_results(j)
        dfs.append(df)
    
    return dfs

df = load_ego_comments()
queries = cocomments_queries(df)
query = queries[0]

def save_all_comments(dfs):
    full_df = pd.concat(dfs)
    full_df.to_pickle('all-cocomments-17-06.pkl')
    full_df.to_csv('all-cocomments-17-06.csv')


full_df.shape


    