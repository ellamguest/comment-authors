#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:16:31 2017

@author: emg
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:36:35 2017

@author: emg
"""

import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import textacy
import spacy
nlp = spacy.load('en')

def basic_df():
    df = pd.read_pickle('cmv_17_06_posts.pkl')
    df['time'] = pd.to_datetime(df['created_utc'], unit='s')
    df['delta'] = df['author_flair_text'].apply(lambda x: re.sub("[^0-9]", "", str(x)))
    df['body_length'] = df['selftext'].apply(lambda x: len(x))
    df['title_length'] = df['title'].apply(lambda x: len(x))
    df = df.replace('', np.nan, regex=True)
    df = df[~df['author'].isin(['[deleted]','AutoModerator'])]
    df['lang'] = df['title'].apply(lambda x: textacy.text_utils.detect_language(x))
    df = df[df['lang']=='en']
    return df

def counts(text):
    exclude = list(string.punctuation)
    sentences = re.split('\n|(?<=\w)[.!?]|\n',text) # split at end punctutation if preceded by alphanumeric
    n_sentences = len([s for s in sentences if s not in [None, '']])
    
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
    text = "".join(ch for ch in text if ch not in exclude)
    n_words = len(text.split(' '))

    if text is None:
        count = 0
    elif len(text) == 0:
        count = 0
    else:
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index-1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        count = count - (0.1*count) # why syllables 0.9 each not 1?
    n_syllables = count

    return n_syllables, n_words, n_sentences

def get_readability_measures(texts):
    '''texts is the pd series (or df column) to perform readabilty measures'''
    df = pd.DataFrame({'text':texts})
    df['doc'] = df['text'].apply(lambda x: textacy.Doc(x.lower()))
    df['word_counts'] = df['doc'].apply(lambda x: x.to_bag_of_words(normalize='lemma', as_strings=True))
    df['n_grams'] = df['doc'].apply(lambda x: list(x.to_terms_list(as_strings=True)))
    
   
    df['n_syllables'], df['n_words'], df['n_sentences'] = list(zip(*
                      df['text'].apply(lambda x: counts(x))))
    
    df['ASL'] = np.divide(df['n_words'], df['n_sentences']) #gives error when dividing by 0
    df['ASW'] = np.divide(df['n_syllables'], df['n_words'])
    
    df['FRE'] = 206.835 - (float(1.015) * df['ASL']) - (float(84.6) * df['ASW'])
    df['FKG'] = (float(0.39) * df['ASL']) + (float(11.8) * df['ASW']) - 15.59
      
    return df
    
def plot(sub, df, x, y, col=None):
    if col==None:
        plt.scatter(x=df[x], y=df[y])
        plt.xlabel(x), plt.ylabel(y)
        plt.title('{} {} by {}'.format(sub, x,y)) 
    else:
        sc = plt.scatter(x=df[x], y=df[y], c=df[col])
        plt.colorbar(sc)
        plt.xlabel(x), plt.ylabel(y)
        plt.title('{} {} by {} (colour = {})'.format(sub, x,y, col))


def double_plot(df1, df2, x, y, cols=['blue','red']):
    plt.scatter(x=df1[x], y=df1[y], c=cols[0], alpha=0.25)
    plt.scatter(x=df2[x], y=df2[y], c=cols[1], alpha=0.25)
    plt.xlabel(x), plt.ylabel(y)
    plt.title('{} by {}'.format(x,y))



plot('CMV posts', read_df, 'FRE', 'FKG')
plot('CMV posts', read_df, 'FRE', 'FKG', col='ASW') # trying to combine plot but not working

## trying to get a way of measuring common terms
def word_freq(word_series):
    all_words = []
    for words in word_series:
        all_words.extend(words)
    all_words.sort()
    freq = [(len(list(group)), key) for key, group in groupby(all_words)]
    freq.sort()
    
    rev_freq = {}
    for key, value in freq:
        rev_freq[value] = key
    
    rev_freq = pd.DataFrame(rev_freq, index=['freq']).T
    return rev_freq

def freq_word_count(word_set, rev_freq):
    n = 0
    for word in word_set:
        n += rev_freq[word]
    return n



### comparing removed posts
def removed(df):
    removed = df[df['selftext']=='[removed]']
    
    return removed

def nonremoved(df):
    nonremoved = df[df['selftext']!='[removed]']
    
    return nonremoved

removed_posts = removed(df)
nonremoved_posts = nonremoved(df)

rem_read = get_readability_measures(removed_posts['title'])
nonrem_read = get_readability_measures(nonremoved_posts['title'])

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.hist(rem_read['n_syllables'], normed=True, label='removed')
ax1.hist(nonrem_read['n_syllables'], normed=True, label='not removed')

def comparison_histogram(df1, df2, variable, labels):
    '''variable is a string of a column in both dfs
       labels is a list of df names'''
    data = [df1[variable], df2[variable]]

    plt.hist(data, label=labels, normed=True)
    plt.title(f'Normalised histograms of {variable} in title')
    plt.legend()
    plt.tight_layout()
    plt.show()

comparison_histogram(rem_read, nonrem_read, 'ASL',
                     ['removed', 'not removed'])





### trying to replace tokenisation w/ textacy lemmatisation

test = df.head()

for title in test['title']:
    # content = 'Certain races are, as a result of genetic factors, at an intellectual disadvantage.'
    doc = textacy.Doc(title.lower())
    #word_counts = doc.to_bag_of_words(normalize='lemma', as_strings=True)
    n_grams = list(doc.to_terms_list(as_strings=True))
    
    print(n_grams)
    print()
    


df = basic_df()
read_df = get_readability_measures(test['title'])
        
dfs = read_df['word_counts']
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df.update(df)

len(merged_df)
