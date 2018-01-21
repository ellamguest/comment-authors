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
import os
import numpy as np
import matplotlib.pyplot as plt
import random


## DATA COLLECTION

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
    authors.sort()

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
    df = pd.read_pickle('1000-random-subs.pkl')
    subreddits = df['subreddit'].unique()
    dfs = []
    for sub in subreddits:
        print(sub)
        df = get_random_authors(sub, n=1000)
        dfs.append(df)
    authors = pd.concat(dfs)
    authors.to_pickle('full-sample.pkl')

def get_all_comments():
    '''
    of the set of 1000 random authors per 1000 random subreddits
    '''
    full = pd.read_pickle('full-sample.pkl')
    qs = separate_queries(full, parts=100)

    for n in range(len(qs)):
        print(n)
        query = qs[n]
        j = run_job(query)
        df = unpack_results(j)
        df.to_pickle('/Users/emg/Programming/GitHub/comment-authors/data/full-random-sample-part-{}.pkl'.format(n))



## DATA COMPILATION AND ANALYSIS

def compile_full_random_sample():
    dfs = []
    print('Compiling full comment set of random sample')
    for filename in os.listdir('/Users/emg/Programming/GitHub/comment-authors/data'):
        df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/{}'.format(filename))
        dfs.append(df)
    
    return pd.concat(dfs)

def get_author_subreddit_counts():
    main = pd.read_pickle('full-sample.pkl')
    for n in range(0,100):
        df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/full-random-sample-part-{}.pkl'.format(n))
        grouped = df.groupby(['author','subreddit']).count()['created_utc']
        grouped_df = pd.DataFrame(grouped).reset_index()
        print('Updating main with subset', n) # this is the slow bit, try dict instead?
        main = (main.merge(grouped_df, how='outer', on = ['author','subreddit'])
                    .rename(columns={'created_utc':n}))
    #main.to_pickle('author_counts.pkl')

    ns = []
    for n in range(0,100):
        ns.append(n)
    
    counts = main[ns].sum(axis=1)
    counts = main[ns].sum(axis=1)
    df = main[['subreddit','author']]
    df['count'] = counts
    
    #df.to_pickle('author_subreddit_comment_counts.pkl')

def get_insubreddt_count(df, sub):
    insub = df[df['subreddit']==sub].groupby('author').count()['count']
    outsub = df[df['subreddit']!=sub].groupby('author').count()['count']
    result = pd.DataFrame({'count':insub,'outcount':outsub,'subreddit':sub}, index=insub.index)
    result.fillna(value=0, inplace=True)
    result['total'] =  result['count'] + result['outcount']
    result['ratio'] = result['count'] / result['total']
    
    return result


def save_subreddit_author_counts():
    df = pd.read_pickle('author_subreddit_comment_counts.pkl')
    pairs = pd.read_pickle('full-sample.pkl')
    subs = pairs['subreddit'].unique()
    subs.sort()
    
    for sub in subs:
        print('Subsetting for', sub)
        authors = pairs[pairs['subreddit']==sub]['author']
        subset = df[df['author'].isin(authors)] # would using groupby be faster?
        insub = get_insubreddt_count(subset, sub)
        
        insub.to_pickle('/Users/emg/Programming/GitHub/comment-authors/data/incounts/{}-author-counts.pkl'.format(sub))
    
def compile_subreddit_author_counts():
    dfs = []
    fs = os.listdir('/Users/emg/Programming/GitHub/comment-authors/data/incounts')
    for filename in fs:
        df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/incounts/{}'.format(filename))
        dfs.append(df)
    
    all_incounts = pd.concat(dfs)
    all_incounts.to_pickle('all_insub_ratios.pkl')

def scatter(x,y):
    fig, ax = plt.subplots()
    ax.scatter(x,y)

def comparision_scatter(x,y):
    xarray, yarray = desc_stats[x], desc_stats[y]
    cmvx, cmvy = cmv_desc[x], cmv_desc[y]
    tdx, tdy = td_desc[x], td_desc[y]
    
    fig, ax = plt.subplots()
    ax.scatter(xarray,yarray,marker='.',alpha=0.5)
    ax.plot(cmvx, cmvy, 'go')
    ax.plot(tdx, tdy, 'ro')
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('{} by {} of in-subreddit author ratios'.format(x,y))
    
def drop_values(df, attribute,minimum):
    return df[df[attribute]>minimum]


df = pd.read_pickle('all_insub_ratios.pkl')
log_ratio = lambda df: np.log(df['count']+1)/np.log(df['total']+1)
df['log_ratio'] = log_ratio(df)


# COMPARING TO REFERNCE SUBREDDITS
cmv_df = pd.read_pickle('cmv_df.pkl')
cmv_ratios = get_insubreddt_count(cmv_df, 'changemyview')

td_df = pd.read_pickle('td_df.pkl')
td_ratios =get_insubreddt_count(td_df,'The_Donald')

for dataset in [df, cmv_ratios, td_ratios]:
    dataset = drop_values(dataset, 'total', 5)
desc_stats = df.groupby('subreddit')['ratio'].describe()
cmv_desc = cmv_ratios['ratio'].describe()
td_desc = td_ratios['ratio'].describe()

comparision_scatter('mean','std')




x, y =  df.groupby('subreddit').mean()['ratio'], np.log(df.groupby('subreddit').mean()['count'])
scatter(x,y)




cmv_ratios['ratio'].std()
td_ratios['ratio'].mean()

cmv_ratios['ratio'].describe()
td_ratios['ratio'].describe()

def ratio_histogram(df, subname):
    fig, ax = plt.subplots()
    ax.hist(df['ratio'])
    ax.set_title('{} Author In-subreddit Ratios Histogram'.format(subname))
    plt.show()

ratio_histogram(td_ratios, 'The_Donald')
ratio_histogram(cmv_ratios, 'changemyview')

def random_ratio_histograms(n=5):
    df = pd.read_pickle('all_insub_ratios.pkl')
    pairs = pd.read_pickle('full-sample.pkl')
    for sub in all_incounts['subreddit'].sample(n):
        authors = pairs[pairs['subreddit']==sub]['author']
        subset = df[df['author'].isin(authors)] # would using groupby be faster?
        ratio_df = get_insubreddt_count(subset, sub)
        
        ratio_histogram(ratio_df, sub)

random_ratio_histograms()


 

    

'''
new_bots = ['ImagesOfNetwork','autotldr','MTGCardFetcher','Roboragi','Smartstocks',
            'TwitterToStreamable', 'MovieGuide', 'SteamKiwi','Mentioned_Videos']

confirmed_not_bots = 'grrrrreat, Pawpatrolbatman, smarvin6689, False1512 piyushsharma301' # many posting in r/counts

now_suspended = ['DemonBurritoCat','strawberrygirl1000']

page_not_found['Not_Just_You']
'''

cmv_df


