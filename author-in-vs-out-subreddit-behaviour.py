#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:45:03 2018

@author: emg
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import Counter

def botless_comments(df, conservative=False):
    '''conservative=True if removing author names containing "bot" '''
    if conservative == True:
        botnames = []
        for name in set(df['author']):
            if 'bot' in name.lower():
                botnames.append(name)
        df = df[~df['author'].isin(botnames)].copy()
    
    remove = ['[deleted]','AutoModerator', 'WikiTextBot','teddyRbot', 'TotesMessenger',
              'Capital_R_and_U_Bot', 'DeltaBot', 'video_descriptionbot', '_youtubot_',
              'gifv-bot','RemindMeBot','timezone_bot','protanoa_is_gay','alotabot',
              'morejpeg_auto', 'ImagesOfNetwork', 'imguralbumbot', 'autotldr','MTGCardFetcher',
              'sneakpeekbot', 'youtubefactsbot', 'Shark_Bot', 'SnapshillBot', 'DemonBurritoCat',
              'HelperBot_', 'cheer_up_bot','anti-gif-bot']
   
    return df[~df['author'].isin(remove)].copy()
    
    

'''
somehow_not_bots = ['grrrrreat', 'Pawpatrolbatman']
'''

def compile_full_random_sample():
    dfs = []
    print('Compiling full comment set of random sample')
    for filename in tqdm(os.listdir('/Users/emg/Programming/GitHub/comment-authors/data')):
        if filename.startswith('full-random-sample-part-'):
            df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/{}'.format(filename))
            dfs.append(df)
    
    return pd.concat(dfs)


def get_random_authors(df, n=100):
    sample_names = []
    subs = df['subreddit'].unique()
    for sub in tqdm(subs):
        subset = df[df['subreddit']==sub]
        random_names = subset['author'].sample(100, replace=True)
        sample_names.extend(random_names)
    return df[df['author'].isin(set(sample_names))]

def comparision_scatter(x,y, main_df, sub_df1, sub_df2):
    xarray, yarray = main_df[x], main_df[y]
    cmvx, cmvy = sub_df1[x], sub_df1[y]
    tdx, tdy = sub_df2[x], sub_df2[y]
    
    fig, ax = plt.subplots()
    ax.scatter(xarray,yarray,marker='.',alpha=0.5, label='random subs')
    ax.plot(cmvx, cmvy, 'go', label = 'changemyview')
    ax.plot(tdx, tdy, 'ro', label = 'The_Donald')
    ax.legend()
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('{} by {} of author subreddit counts'.format(x,y))

def update_counts(old_d, new_d):
    return dict(Counter(old_d) + Counter(new_d))

def get_comment_counts():
    pairs = pd.read_pickle('full-sample.pkl')
    pairs = botless_comments(pairs)
    
    a_counts, s_counts, a_s_counts = {}, {}, {}
    for i in tqdm(range(0,100)):
        partial_df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/full-random-sample-part-{}.pkl'.format(i))
        new_a_counts = partial_df.groupby('author')['created_utc'].count().to_dict()
        new_s_counts = partial_df.groupby('subreddit')['created_utc'].count().to_dict()
        new_a_s_counts = partial_df.groupby(['author', 'subreddit'])['created_utc'].count().to_dict()
        
        a_counts = Counter(a_counts) + Counter(new_a_counts)
        s_counts = Counter(s_counts) + Counter(new_s_counts) 
        a_s_counts = Counter(a_s_counts) + Counter(new_a_s_counts) 
        
    pairs['author_comment_count'] = pairs['author'].map(lambda x: a_counts.get(x, np.nan))
    pairs['subreddit_comment_count'] = pairs['subreddit'].map(lambda x: s_counts.get(x, np.nan))
    pairs['author_subreddit_comment_count'] = partial_df.head().apply(
            lambda row: a_s_counts[(row['author'], row['subreddit'])], axis=1)
    
    sub_author_counts = pairs.drop_duplicates(['subreddit','author']).groupby('subreddit')['author'].count().to_dict()
    pairs['sub_author_count'] = pairs['subreddit'].map(lambda x: sub_author_counts[x])
    
    pairs.to_pickle('full_sample_comment_counts.pkl')

def get_subreddit_count_stats(df, count_type):
    ''' df should be pairs
        count_type should be 'author_comment_count', 'subreddit_comment_count',
        or 'author_comment_count'
    '''
    stats = {}
    subs = df['subreddit'].unique()
    for sub in tqdm(subs):
        stats[sub] = df[df['subreddit']==sub][count_type].describe()
    
    return pd.DataFrame.from_dict(stats, orient='index')

### get count stats for control subreddits
pairs = pd.read_pickle('full_sample_comment_counts.pkl')
pairs = botless_comments(pairs)
a_counts = get_subreddit_count_stats(pairs,'author_comment_count')
s_counts = get_subreddit_count_stats(pairs,'subreddit_comment_count')
a_s_counts = get_subreddit_count_stats(pairs,'author_subreddit_comment_count')

a_counts.to_pickle('author_comment_count_stats_by_author.pkl')
s_counts.to_pickle('subreddit_comment_count_stats_by_author.pkl')
a_s_counts.to_pickle('author_local_comment_counts_by_subreddit.pkl')

full_stats = {}
full_stats['author_comment_count'] = a_counts
full_stats['subreddit_comment_count'] = s_counts
full_stats['author_subreddit_comment_count'] = a_s_counts


### get count stats for target subreddits
cmv_df = pd.read_pickle('cmv_df.pkl').pipe(botless_comments)
td_df = pd.read_pickle('td_df.pkl').pipe(botless_comments)

def count_stats(df):
    d = {}
    d['author_comment_count'] = df.groupby('author')['created_utc'].count().describe()
    d['subreddit_comment_count'] = df.groupby('subreddit')['created_utc'].count().describe()
    d['author_subreddit_comment_count'] =  df.groupby(['author', 'subreddit'])['created_utc'].count().describe()
    
    return d

c_stats = count_stats(cmv_df)
t_stats = count_stats(td_df)



a_counts = pd.read_pickle('author_comment_count_stats_by_author.pkl')
s_counts = pd.read_pickle('subreddit_comment_count_stats_by_author.pkl')
a_s_counts = pd.read_pickle('author_local_comment_counts_by_subreddit.pkl')

a_counts = pd.read_pickle('author_comment_count_stats_by_author.pkl')



def get_sub_author_counts(df, level=''):
    '''gets the number o samples authors for each subreddit'''
    d = df.drop_duplicates(['subreddit','author']).groupby('subreddit')['author'].count().to_dict()
    df['{}sub_author_count'.format(level)] = df['subreddit'].map(lambda x: d[x])
    return df

org_pairs = pd.read_pickle('full-sample.pkl')
org_pairs = get_sub_author_counts(org_pairs, level='org_')

botless_pairs = botless_comments(org_pairs)
botless_pairs = get_sub_author_counts(botless_pairs, level='botless_')

con_botless_pairs = botless_comments(botless_pairs, conservative=True)
con_botless_pairs = get_sub_author_counts(con_botless_pairs, level='con_botless_')


subset = con_botless_pairs.drop_duplicates('subreddit')
subset['removed_bots'] = subset['org_sub_author_count'] - subset['botless_sub_author_count']
subset['removed_con_bots'] = subset['botless_sub_author_count'] - subset['con_botless_sub_author_count']



fig, ax = plt.subplots()
ax.scatter(subset['org_sub_author_count'], subset['botless_sub_author_count'])





pairs = pd.read_pickle('full_sample_comment_counts.pkl')
pairs = botless_comments(pairs)
s_counts = get_subreddit_count_stats(pairs,'author_comment_count')




get author sub count dictionary
apply to authors in pairs
breakdown pairs into stats


def compile_full_random_sample2():
    dfs = []
    print('Compiling full comment set of random sample')
    for filename in tqdm(os.listdir('/Users/emg/Programming/GitHub/comment-authors/data')):
        if filename.startswith('full-random-sample-part-'):
            df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/{}'.format(filename))
            df = df.drop_duplicates(['subreddit','author'])
            dfs.append(df)
    
    return pd.concat(dfs)

def get_author_sub_counts(df, level=''):
    '''gets the number o samples authors for each subreddit'''
    d = df.drop_duplicates(['subreddit','author']).groupby('subreddit')['author'].count().to_dict()
    df['{}sub_author_count'.format(level)] = df['subreddit'].map(lambda x: d[x])
    return df

dataset = compile_full_random_sample2()
author_subreddit_count_dict = dataset.groupby('author')['subreddit'].count().to_dict()
pairs = pd.read_pickle('full_sample_comment_counts.pkl')
#pairs = botless_comments(pairs)
pairs['author_sub_count'] = pairs['author'].map(lambda x: author_subreddit_count_dict[x])

sub_author_sub_counts = pairs.groupby('subreddit')['author_sub_count'].describe()
subset = a_counts[a_counts['count']>980]

def get_random_sample_authors(ego_df):
    df = botless_comments(ego_df)
    names = np.random.choice(df['author'].unique(), 1000)
    sample = df[df['author'].isin(names)]
    
    return sample

cmv_sample = get_random_sample_authors(cmv_df)
c_stats = cmv_sample.groupby('author')['subreddit'].count().describe()
c_stats
medians.append(c_stats['50%'])

cmv_df = pd.read_pickle('cmv_df.pkl')
cmv_df = botless_comments(cmv_df, conservative=True)
c_stats = cmv_df.groupby('author')['subreddit'].count().describe()

td_df = pd.read_pickle('td_df.pkl')
td_df = botless_comments(td_df, conservative=True)
td_sample = get_random_sample_authors(td_df)
t_stats = td_sample.groupby('author')['subreddit'].count().describe()


def comparision_scatter(sample_array, cmv_stats, td_stats, x,y):
    xarray, yarray = np.log(sample_array[x]), np.log(sample_array[y])
    cmvx, cmvy = np.log(cmv_stats[x]), np.log(cmv_stats[y])
    tdx, tdy = np.log(td_stats[x]), np.log(td_stats[y])
    
    fig, ax = plt.subplots()
    ax.scatter(xarray,yarray,marker='.',alpha=0.5, label='random subs')
    ax.plot(cmvx, cmvy, 'go', label = 'changemyview')
    ax.plot(tdx, tdy, 'ro', label = 'The_Donald')
    ax.legend()
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title('{} by {} of {}'.format(x,y, count_type))

subset = sub_author_sub_counts[sub_author_sub_counts['count']>980]    
comparision_scatter(subset, c_stats, t_stats, 'mean','50%')


fig, ax = plt.subplots()
ax.hist(subset['50%'])
ax.set_title('Histogram of Median Alter-Subreddit Counts by Ego-Subreddit')
ax.set_xlabel('Median Number Alter-Subreddits by Authors')
ax.set_ylabel('Count of Ego-Subreddits')
plt.tight_layout()

