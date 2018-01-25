#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:05:11 2018

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
    
   
def compile_full_random_sample2():
    dfs = []
    print('Compiling full comment set of random sample')
    for filename in tqdm(os.listdir('/Users/emg/Programming/GitHub/comment-authors/data')):
        if filename.startswith('full-random-sample-part-'):
            df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/{}'.format(filename))
            df = df.drop_duplicates(['subreddit','author'])
            dfs.append(df)
    
    return pd.concat(dfs)



def plotsub_count_hist():
    fig, ax = plt.subplots()
    ax.hist(subset['50%'])
    ax.set_title('Histogram of Median Alter-Subreddit Counts by Ego-Subreddit')
    ax.set_xlabel('Median Number Alter-Subreddits by Authors')
    ax.set_ylabel('Count of Ego-Subreddits')
    plt.tight_layout()

dataset = compile_full_random_sample2()
author_subreddit_count_dict = dataset.groupby('author')['subreddit'].count().to_dict()
pairs = pd.read_pickle('full_sample_comment_counts.pkl')
#pairs = botless_comments(pairs)
pairs['author_sub_count'] = pairs['author'].map(lambda x: author_subreddit_count_dict[x])

sub_author_sub_counts = pairs.groupby('subreddit')['author_sub_count'].describe()
subset = sub_author_sub_counts[sub_author_sub_counts['count']>980]    

cmv_df = pd.read_pickle('cmv_df.pkl')
cmv_df = botless_comments(cmv_df, conservative=True)
cmv_sample = get_random_sample_authors(cmv_df)
#c_stats = cmv_sample.groupby('author')['subreddit'].count().describe()
c_stats = cmv_df.groupby('author')['subreddit'].count().describe()

td_df = pd.read_pickle('td_df.pkl')
td_df = botless_comments(td_df, conservative=True)
td_sample = get_random_sample_authors(td_df)
#t_stats = td_sample.groupby('author')['subreddit'].count().describe()
t_stats = td_df.groupby('author')['subreddit'].count().describe()



def log_comparision_scatter(sample_array, cmv_stats, td_stats, x,y):
    xarray, yarray = np.log(sample_array[x]), np.log(sample_array[y])
    cmvx, cmvy = np.log(cmv_stats[x]), np.log(cmv_stats[y])
    tdx, tdy = np.log(td_stats[x]), np.log(td_stats[y])
    
    fig, ax = plt.subplots()
    ax.scatter(xarray,yarray,marker='.',alpha=0.5, label='random subs')
    ax.plot(cmvx, cmvy, 'go', label = 'changemyview')
    ax.plot(tdx, tdy, 'ro', label = 'The_Donald')
    ax.legend()
    
    ax.set_xlabel('Log Std Dev Author Alter-Subreddit Count')
    ax.set_ylabel('Log Mean of Author Alter-Subreddit Count')
    ax.set_title('Alter-Subreddit Count Stats by Subreddit')
    
    plt.tight_layout()

def comparision_scatter(sample_array, cmv_stats, td_stats, x,y):
    xarray, yarray = sample_array[x], sample_array[y]
    cmvx, cmvy = cmv_stats[x], cmv_stats[y]
    tdx, tdy = td_stats[x], td_stats[y]
    
    fig, ax = plt.subplots()
    ax.scatter(xarray,yarray,marker='.',alpha=0.5, label='random subs')
    ax.plot(cmvx, cmvy, 'go', label = 'changemyview')
    ax.plot(tdx, tdy, 'ro', label = 'The_Donald')
    ax.legend()
    
    #ax.set_xlabel('Std Dev Author Alter-Subreddit Count')
    #ax.set_ylabel('Mean of Author Alter-Subreddit Count')
    #ax.set_title('Alter-Subreddit Count Stats by Subreddit')
    
    plt.tight_layout()
    
comparision_scatter(subset, c_stats, t_stats, 'std','mean')


### 3.2 RATIO OF IN-SUBREDDIT VS OUT-SUBREDDIT AUTHOR ENGAGEMENT
pairs['insub_ratio'] = pairs['author_subreddit_comment_count'] / pairs['author_comment_count']
insub_ratios = pairs.groupby('subreddit')['insub_ratio'].describe()
#in_subset = insub_ratios[insub_ratios['count']>980] 

comparision_scatter(insub_ratios, c_ratio, t_ratio, '50%','mean')


def get_insubreddit_count(df, sub):
    insub = df[df['subreddit']==sub].groupby('author')['subreddit'].count()
    total = df.groupby('author')['subreddit'].count()
    return insub/total



c_ratio = get_insubreddit_count(cmv_df, 'changemyview').describe()
t_ratio = get_insubreddit_count(td_df, 'The_Donald').describe()


df = pd.read_pickle('all_insub_ratios.pkl')
df2 = botless_comments(df)
insub_ratios = df2.groupby('subreddit')['ratio'].describe()


fig, ax = plt.subplots()
ax.hist(insub_ratios['50%'])
ax.set_title('Histogram of Median Insubbredit Ratios by Ego-Subreddit')
ax.set_xlabel('Median In-subreddit Ratio by Authors')
ax.set_ylabel('Count of Ego-Subreddits')
plt.tight_layout()

