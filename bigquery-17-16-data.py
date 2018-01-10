#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:18:27 2018

@author: emg
"""

import pandas as pd
from tqdm import tqdm
from google.cloud import bigquery
from concurrent.futures import TimeoutError


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

def separate_queries(df, parts=4):
    '''full list of authors exceeds query character length so divide into sublists'''
    authors = list(set(df['author']))

    sublists = split_list(authors, parts)

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

def all_cocomments(df):
    queries = separate_queries(df)
    dfs = []
    for query in queries:
        j = run_job(query)
        print()
        df = unpack_results(j)
        dfs.append(df)
    
    return dfs

'''df = load_ego_comments()
queries = cocomments_queries(df)
query = queries[0]
'''

def save_all_comments(dfs):
    full_df = pd.concat(dfs)
    full_df.to_pickle('all-cocomments-17-06.pkl')
    full_df.to_csv('all-cocomments-17-06.csv')


def COMPLETE_DATASET():
    '''
    queries quickly but unpacking lags indefinitely
    '''
    config.maximum_bytes_billed = int(22e9)
    COMPLETE_QUERY = """SELECT subreddit, author, created_utc
                        FROM `fh-bigquery.reddit_comments.2017_06`"""
    j = run_job(COMPLETE_QUERY)
    print()
    df = unpack_results(j)
    
    df.to_pickle('COMPLETE-COMMENT-DATASET-17-06.pkl')
    
'''
REFERENCE SAMPLE DATASET
- take 1000 random subreddits (excluding defaults)
- take 1000 random authors from each subreddit
- get all comments by those authors across all subreddits
- do in-subreddit author ratios for each subreddit

TEST SET
- take 10 random subreddits (excluding defaults)
- take 100 authors from each subreddit
- compile list of all authors
- collect all comments by those authors
- do ratios
'''

def get_random_subreddits(n=1000):
    defaults_list = 'Art+AskReddit+DIY+Documentaries+EarthPorn+Futurology+GetMotivated+IAmA+InternetIsBeautiful+Jokes+LifeProTips+Music+OldSchoolCool+Showerthoughts+UpliftingNews+announcements+askscience+aww+blog+books+creepy+dataisbeautiful+explainlikeimfive+food+funny+gadgets+gaming+gifs+history+listentothis+mildlyinteresting+movies+news+nosleep+nottheonion+personalfinance+philosophy+photoshopbattles+pics+science+space+sports+television+tifu+todayilearned+videos+worldnews'
    defaults = defaults_list.replace('+', "', '")
    
    query = """SELECT DISTINCT subreddit
                FROM `fh-bigquery.reddit_comments.2017_06`
                WHERE subreddit NOT IN ('{}')
                LIMIT {}""".format(defaults, n)
    
    j = run_job(query)
    df = unpack_results(j)
    df.to_pickle('{}-random-subs.pkl'.format(n))
    
    return df

def get_random_authors(subreddits, n=1000):
    '''
    N.B. skipping all authors w/ 'bot' in name and known bots w/o 'bot in name
    '''
    subnames = ''
    for name in subreddits:
        subnames = subnames+ "'" + name + "' ,"
    
    subnames= subnames[:-2]
    
    skip = "'[deleted]','AutoModerator', 'TotesMessenger', 'protanoa_is_gay', 'morejpeg_auto'"
        
    query = """SELECT DISTINCT subreddit, author
                FROM `fh-bigquery.reddit_comments.2017_06`
                WHERE subreddit IN ({}) AND
                    LOWER(author) NOT LIKE 'bot' AND
                     author NOT IN ({})
                LIMIT {}""".format(subnames, skip, n)
 
    j = run_job(query)
    output = unpack_results(j)
    output.to_pickle('{}-random-authors.pkl'.format(n))
    return output

df = get_random_subreddits()
random_subs = pd.read_pickle('1000-random-subs.pkl') #1000 random subreddit entries, 612 unique subreddits
subreddits = random_subs['subreddit'].unique()
random_authors = get_random_authors(subreddits)

x = pd.read_pickle('all-authors-random-subs.pkl')

random_authors = get_random_authors(subreddits[:5], n=50)

def get_random_authors(subreddit, n=1000):
    '''
    N.B. skipping all authors w/ 'bot' in name and known bots w/o 'bot in name
    *** NOT SKIPPING BOTS YET
    '''
    skip = "'[deleted]','AutoModerator', 'TotesMessenger', 'protanoa_is_gay', 'morejpeg_auto'"
        
    query = """SELECT DISTINCT subreddit, author
                FROM `fh-bigquery.reddit_comments.2017_06`
                WHERE subreddit = '{}' AND
                    LOWER(author) NOT LIKE 'bot' AND
                     author NOT IN ({})
                LIMIT {}""".format(subreddit, skip, n)
 
    j = run_job(query)
    df = unpack_results(j)
    return df

def get_many_authors():
    subreddits = random_subs['subreddit'].unique()
    dfs = []
    for sub in subreddits:
        print(sub)
        df = get_random_authors(sub, n=1000)
        dfs.append(df)
    full = pd.concat(dfs)
    full.to_pickle('full-sample.pkl')

full = pd.read_pickle('full-sample.pkl')
names = full['author'].unique()




qs = separate_queries(full, parts=100)



