#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:02:01 2017

@author: emg
"""
import pandas as pd
import re
import numpy as np

df = pd.read_pickle('cmv_17_06_posts.pkl')
df['time'] = pd.to_datetime(df['created_utc'], unit='s')
df['delta'] = df['author_flair_text'].apply(lambda x: re.sub("[^0-9]", "", str(x)))
df['body_length'] = df['selftext'].apply(lambda x: len(x))
df['title_length'] = df['title'].apply(lambda x: len(x))
df = df.replace('', np.nan, regex=True)

subset = df[~df['author'].isin(['[deleted]','AutoModerator'])]
subset = subset[~subset['selftext'].isin(['[deleted]','[removed]', '[]'])]

def stats_table(df):
    df_stats = pd.DataFrame(
            {'score':df.score.describe(), 
             'num_comments':df.num_comments.describe(),
             'delta':df.delta.describe(),
             'body_length':df.body_length.describe(),
             'title_length':df.title_length.describe()}
            )
    
    return df_stats

### AUTHOR FREQUENCY
author_counts = subset.author.value_counts()

single_authors = author_counts[author_counts==1].index
singles = subset[subset['author'].isin(single_authors)]

repeats = subset[~subset['author'].isin(single_authors)]
    

### REPOSTING
def removed(df):
    removed = df[df['selftext']=='[removed]']
    
    return removed

def nonremoved(df):
    nonremoved = df[df['selftext']!='[removed]']
    
    return nonremoved

title_counts = df.title.value_counts()
repeat_titles = df[df['title'].isin(
        title_counts[title_counts!=1].index)]
repeat_titles = repeat_titles.copy().sort_values(['title','created_utc','author'])

unique_titles = repeat_titles.title.unique().shape[0]

firsts = repeat_titles.drop_duplicates('title', keep='first')
assert (firsts.shape[0] == unique_titles), '# firsts don\'t match # repeats' 

lasts = repeat_titles.drop_duplicates('title', keep='last')
assert (lasts.shape[0] == unique_titles), '# lasts don\'t match # repeats' 


# repeated titles that are re-posted by different authors
unoriginal_repeats = repeat_titles.drop_duplicates(['title','author'],keep=False)
unoriginal_repeats = repeat_titles[repeat_titles.duplicated(['title','author'],keep=False)==False]

unoriginal_firsts = unoriginal_repeats.drop_duplicates(['title','author'], keep='first')
unoriginal_lasts = unoriginal_repeats.drop_duplicates(['title','author'], keep='last')

assert(unoriginal_firsts.shape[0] + unoriginal_lasts.shape[0]
         <= unoriginal_repeats.shape[0]), 'more unorg firsts + lasts than total unorg repeats'



# repeat titles that are re-posted by the same author
original_repeats = repeat_titles[repeat_titles.duplicated(['title','author'],keep=False)==True]

assert (original_repeats.shape[0] + unoriginal_repeats.shape[0] 
        == repeat_titles.shape[0]), 'originals plus unoriginals != # repeats'

original_firsts = original_repeats.drop_duplicates(['title','author'], keep='first')
original_lasts = original_repeats.drop_duplicates(['title','author'], keep='last')

assert(original_firsts.shape[0] + original_lasts.shape[0]
         <= original_repeats.shape[0]), 'more og firsts + lasts than total og repeats'


removed_originals = removed(original_repeats)
rem_og_firsts = removed(original_firsts)
rem_og_lasts = removed(original_lasts)


subsets = [repeat_titles,firsts, lasts, unoriginal_repeats,
          original_repeats, original_firsts, original_lasts]
names = ['all_repeat_titles','all_repeat_firsts','all_repeat_lasts','unoriginal_repeats',
         'original_repeats', 'original_firsts', 'original_lasts']

def print_statements(subsets, names):
    n = 0
    for subset in subsets:
        print()
        print(f'there are {len(removed(subset))} removed {names[n]}'
              f', or {len(removed(subset))/len(subset):.2%} of all {names[n]}')
        
        n += 1
    

print_statements(subsets, names)

def removed_counts_df(subsets, names):  
    n = 0
    d = {}
    for subset in subsets:
        d[names[n]] = [len(subset), len(removed(subset))]
        n += 1
    
    repeats_df = pd.DataFrame(d, index = ['total','removed'], columns = names).T
    repeats_df['rate_removed'] = repeats_df['removed']/repeats_df['total']
    
    return repeats_df

repeats_df = removed_counts_df(subsets, names)
repeats_df.sort_values('rate_removed', ascending=False)


### textacy.emotional valence?
import textacy
import spacy
nlp = spacy.load('en')

title = repeat_titles['title'].iloc[10]
words = nlp(title)

words

def emo_scores(title):
    words = nlp(title)
    emo_scores = textacy.lexicon_methods.emotional_valence(words)
    
    return emo_scores

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

def get_emo_df(df):
    emo_dict = df['title'].apply(lambda x: emo_scores(x))
    emo_df = pd.DataFrame(list(emo_dict), index=list(df['title']))
    emo_df['title'] = emo_df.index
    emo_df['polarity'], emo_df['subjectivity'] = list(zip(*emo_df['title'].apply(lambda x: get_sentiment(x))))
    
    return emo_df

test = df.head()   
emo_df = get_emo_df(df)


emo_df.sort_values('polarity')[['polarity', 'subjectivity']]

from textblob import TextBlob

blob = TextBlob('US should NOT pull out of the Paris agreement')



for title in emo_df[emo_df['polarity']==-1]['title']:
    print(title)
    print()






