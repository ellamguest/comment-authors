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

def botless_comments(df):
    remove = ['[deleted]','AutoModerator', 'WikiTextBot','teddyRbot', 'TotesMessenger',
              'Capital_R_and_U_Bot', 'DeltaBot', 'video_descriptionbot', '_youtubot_',
              'gifv-bot','RemindMeBot','timezone_bot','protanoa_is_gay','alotabot',
              'morejpeg_auto']
    return df[~df['author'].isin(remove)].copy()

cmv_df = pd.read_pickle('cmv_df.pkl')
cmv_df = botless_comments(cmv_df)
#c_sample_names = np.random.choice(cmv_df['author'].unique(),size=1000, replace=False)
#c_sample = cmv_df[cmv_df['author'].isin(c_sample_names)]

td_df = pd.read_pickle('td_df.pkl')
td_df = botless_comments(td_df)
#t_sample_names = np.random.choice(td_df['author'].unique(),size=1000, replace=False)
#t_sample = td_df[td_df['author'].isin(t_sample_names)]

count_stats = lambda x: x.groupby('author')['subreddit'].count().describe()

#c_sample_stats = count_stats(c_sample)
c_stats = count_stats(cmv_df)

#t_sample_stats = count_stats(t_sample)
t_stats = count_stats(td_df)


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


#df = compile_full_random_sample()
#df = botless_comments(df)
pairs = pd.read_pickle('full-sample.pkl')
pairs = botless_comments(pairs)
grouped = pairs.groupby('subreddit')
sample = grouped['author'].apply(lambda x: x.sample(10, replace=True))
sample.shape


def update_counts(old_d, new_d):
    return dict(Counter(old_d) + Counter(new_d))

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

author_comment_df = {}
for i in tqdm(range(0,100)):
    partial_df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/full-random-sample-part-{}.pkl'.format(i))
    old_d = dict(author_comment_df)
    new_d = partial_df.groupby('author')['subreddit'].count().to_dict()
    author_comment_df = Counter(old_d) + Counter(new_d)

stats = {}
subs = pairs['subreddit'].unique()
for sub in tqdm(subs):
    stats[sub] = pairs[pairs['subreddit']==sub]['count'].describe()

stats_df = pd.DataFrame.from_dict(stats, orient='index')
active = stats_df[stats_df['count']>99]


comparision_scatter('count','50%',active,c_stats,t_stats)

# getting author comment counts by subreddit
test = 




