# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

remove = ['[deleted]','AutoModerator', 'WikiTextBot','teddyRbot', 'TotesMessenger',
          'Capital_R_and_U_Bot']
def remove_authors(df, remove=remove):
    return df[~df['author'].isin(remove)].copy()

print('Opening full comment files...')
cmv = pd.read_pickle('cmv-17-06-cocomments.pkl')
cmv = remove_authors(cmv)
td = pd.read_pickle('td-17-06-cocomments.pkl')
td = remove_authors(td)

'''print('Opening ego comment files...')
cmv_comments = pd.read_csv('cmv_sample.csv')
cmv_comments = remove_authors(cmv_comments)

td_comments = pd.read_csv('td_sample.csv')
td_comments = remove_authors(td_comments)
'''

def sub_counts(df):
    counts = df['subreddit'].value_counts()
    sub_counts = pd.DataFrame(sub_counts).rename({'subreddit':'count'})
    sub_counts['portion'] = sub_counts['subreddit'] / sub_counts['subreddit'].sum()
    author_counts = pd.DataFrame({'count':counts,
                                  'portion':counts/df.shape[0]})
    
    
    return sub_counts
    
cmv_counts = sub_counts(cmv)
td_counts = sub_counts(td)

def count_df(series):
    '''series is the df column of values to count'''
    counts = series.value_counts()
    counts_df = pd.DataFrame({'count':counts,
                                  'portion':counts/series.shape[0]})
    return counts_df

cmv_authors = count_df(cmv['author'])
cmv_subreddits = count_df(cmv['subreddit'])

td_authors = count_df(td['author'])
td_subreddits = count_df(td['subreddit'])

cmv_matrix = cmv.pivot_table(values='count',index='author',columns='subreddit',aggfunc='sum', fill_value=0)

print('CMV authors posted in {} other subreddits'.format(cmv_counts.shape[0]))
print('TD authors posted in {} other subreddits'.format(td_counts.shape[0]))

cmv_counts.sort_values('subreddit').tail()
td_counts.sort_values('subreddit').tail()

print('Prepping data...')
cmv_names = set(cmv['author'])
td_names = set(td['author'])
cmv_subs = set(cmv['subreddit'])
td_subs = set(td['subreddit'])

shared_authors = cmv_names.intersection(td_names)
shared_subreddits = cmv_subs.intersection(td_subs)

print('CMV and TD share {} authors and {} subreddits'.format(len(shared_authors), len(shared_subreddits)))

cmv_shared = cmv[cmv['author'].isin(shared_authors)]
td_shared = td[td['author'].isin(shared_authors)]

print('Getting union comments...')
union = pd.merge(cmv, td, how='inner')
remove = ['[deleted]','AutoModerator', 'WikiTextBot','teddyRbot']
botless = union[~union['author'].isin(remove)]

print('Done!')