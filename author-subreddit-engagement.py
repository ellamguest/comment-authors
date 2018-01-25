# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import matplotlib.pyplot as plt

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
    
    

def make_subsets(df):
    cmv_authors = df[df['subreddit']=='changemyview']['author'].unique()
    cmv_df = df[df['author'].isin(cmv_authors)]
    
    td_authors = df[df['subreddit']=='The_Donald']['author'].unique()
    td_df = df[df['author'].isin(td_authors)]
    
    return cmv_df, td_df

def df2matrix(df, index, columns):
    '''index and columns are strings of df column names'''
    df['count'] = 1
    matrix = df.pivot_table(values='count',index=index,columns=columns,
                            aggfunc='sum', fill_value=0)
    
    return matrix
    
def save_subsets():
    cmv_df.to_pickle('cmv_df.pkl')
    
    td_df.to_pickle('td_df.pkl')

def load_subsets():
    cmv_df = pd.read_pickle('cmv_df.pkl')
    
    td_df = pd.read_pickle('td_df.pkl')
    
    return cmv_df, td_df

def counts(df):
    d = {'comments':df.shape[0],
         'authors':df.author.value_counts().describe(),
         'subreddits':df.subreddit.value_counts().describe()}
    
    return d

def check_counts():
    cmv_counts = counts(cmv_df)
    td_counts = counts(td_df)
        
    print(f"{cmv_counts['authors']['count']} CMV authors posted \
             {cmv_counts['comments']} comments in \
             {cmv_counts['subreddits']['count']} subreddits")
    
    print(f"{td_counts['authors']['count']} TD authors posted \
             {td_counts['comments']} comments in \
             {td_counts['subreddits']['count']} subreddits")

def shared():
    shared_authors = set(cmv_df['author']).intersection(set(td_df['author']))
    shared_subreddits = set(cmv_df['subreddit']).intersection(set(td_df['subreddit']))
    
    print('CMV and TD share {} authors and {} subreddits'.format(len(shared_authors), len(shared_subreddits)))
    
    shared = df[df['author'].isin(shared_authors)]


'''
analyses to try
- compare rankings of most pop subreddits by td and cmv authors
- t-test on weights?
- range of cmv and td ratios by others


'''
def ranked_df(df, column):
    ranked = pd.DataFrame({'count':df[column].value_counts(),
                           'weighted':df[column].value_counts()/df.shape[0]})
    
    return ranked

def ranked_subsets():
    td_rank = ranked_df(td_df, 'subreddit')
    cmv_rank = ranked_df(cmv_df, 'subreddit')
    
    shared_ranks = pd.merge(td_rank,cmv_rank,'inner', left_index=True, right_index=True)

def get_insubreddt_count(df, sub):
    insub = df[df['subreddit']==sub].set_index('author')
    insub['outcount'] = df[df['subreddit']!=sub]['author'].value_counts(dropna=False)
    insub['outcount']= insub['outcount'].fillna(0)
    insub['total'] = insub['count']+insub['outcount']
    insub['ratio']= insub['count']/insub['total']
    return insub

df = pd.read_pickle('all-cocomments-17-06.pkl')
df = df[df['author']!='[deleted]'] # keeping bots for now
cmv_df = pd.read_pickle('cmv_df.pkl')
td_df = pd.read_pickle('td_df.pkl')
cmv_ratios = get_insubreddt_count(cmv_df, 'changemyview')
td_ratios =get_insubreddt_count(td_df,'The_Donald')


def ratio_histogram(df):
    fig, ax = plt.subplots()
    ax.hist(df['ratio'])
    ax.set_title('TD Author Ratios Histogram')
    
ratio_histogram(td_trim)

td_trim = td_ratios[td_ratios['total']>5]

td_full = td_ratios[td_ratios['ratio']==1]

def ratio_plot(df):
    fig, ax = plt.subplots()
    ax.scatter(df['ratio'],df['total'], s=5)
    #ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
    ax.set_title('ratio by total number of comments by authors')
    ax.set_ylabel('Number of comments')
    ax.set_xlabel('In-subreddit ratio')
    #ax.set_ylim(ymin=0.5)
    #ax.set_xlim(xmin=0.1)

ratio_plot(td_ratios)
