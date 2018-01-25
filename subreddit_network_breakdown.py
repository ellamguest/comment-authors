#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:52:20 2018

@author: emg
"""

import pandas as pd
from tqdm import tqdm
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import matplotlib.pyplot as plt

'''
want to get the author-subreddit network for each random subreddit

a) select node subreddit
b) get list of authors for node subreddit
c) get all comments by authors
d) get author-subreddit matrix
e) do basic network stats (esp density)
f) compare network stats for all node subreddits

ideally would use number of comments as weighted ties
in first instance could just look at unweighted ties

TEST SAMPLE
a) 10 node subreddits
b) 10 authors per subreddit
c) get all comments for that group of ~100 authors (binning to save time iterating by subreddit)
d) then subset by 
'''
def random_sample(df, subreddit_n=100, author_n=100, replace_authors=False):
    sample_names = []
    subs = df['subreddit'].sample(subreddit_n)
    for sub in tqdm(subs):
        subset = df[df['subreddit']==sub]
        random_names = subset['author'].sample(author_n, replace=replace_authors)
        sample_names.extend(random_names)
    return df[df['author'].isin(set(sample_names)) & df['subreddit'].isin(subs)]

def get_sample_comments(df):
    dfs = []
    for i in tqdm(range(0,100)):
            partial_df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/data/full-random-sample-part-{}.pkl'.format(i))
            subset = partial_df[partial_df['author'].isin(sample['author'])]
            dfs.append(subset)
    len(dfs)
    
    return pd.concat(dfs)

def bipartite_net(df):
    set0 = set(df['subreddit'])
    set1 = set(df['author'])
    edges = list(zip(df['subreddit'],df['author']))
    
    B = nx.Graph()
    B.add_nodes_from(set0, bipartite=0)
    B.add_nodes_from(set1, bipartite=1)
    B.add_edges_from(edges)
    
    return B

pairs = pd.read_pickle('full-sample.pkl')
pairs['subreddit'] = 'r/' + pairs['subreddit']
sample = random_sample(pairs, subreddit_n=100, author_n=100, replace_authors=True)
test = get_sample_comments(sample)
test['subreddit'] = 'r/' + test['subreddit']

sub_groups = sample.groupby('subreddit')
author_dict = {}
for sub in tqdm(set(sample['subreddit'])):
    authors = set(sub_groups.get_group(sub)['author'])
    author_dict[sub] = authors    

edgelist_dict = {}
for sub in tqdm(set(sample['subreddit'])):
    authors = author_dict[sub]
    edgelist = test[test['author'].isin(authors)]
    edgelist_dict[sub] = edgelist

net_dict = {}
for sub in tqdm(set(sample['subreddit'])):
    net_dict[sub] = bipartite_net(edgelist_dict[sub])

net_stats = {}
for sub in tqdm(set(sample['subreddit'])):
    B = net_dict[sub]
    if nx.is_bipartite(net_dict[sub]) == True:
        density = nx.density(B)
        bottom_nodes, top_nodes = bipartite.sets(B)
        N_subs, N_authors = len(bottom_nodes), len(top_nodes)
        
        net_stats[sub] = (density, N_subs, N_authors)
    else: # was having trouble when author and subreddit have same name
        net_stats[sub] = (np.nan, np.nan, np.nan)
    
net_stats_df = pd.DataFrame.from_dict(net_stats).T
net_stats_df.columns = ['density','N_subs','N_authors']
net_stats_df.sort_values('density')

fig, ax = plt.subplots()
ax.scatter(net_stats_df['density'], net_stats_df['N_subs'])
ax.set_xlabel('# subreddits in network')
ax.set_ylabel('Network Density')
ax.set_title('Author-Subreddit Network Stats by Node Sub')


