#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:40:09 2017

@author: emg
"""
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
#### comment author by post network

df = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/cmv_17_06_comments.pkl')

edgelist = df[['author','link_id']]

B = nx.Graph()
B.add_nodes_from(set(edgelist['author']), bipartite=0) 
B.add_nodes_from(set(edgelist['link_id']), bipartite=1)
B.add_edges_from(list(zip(edgelist['author'],edgelist['link_id'])))

assert (len(B.nodes()) == len(df['author'].unique()) + len(df['link_id'].unique())), 'number of nodes is off'
assert (len(B.edges()) == len(edgelist.drop_duplicates())), 'number of edges is off'

subgraphs = list(nx.connected_component_subgraphs(B))
giant = sorted(subgraphs, key=len)[-1]

remove = [node for node,degree in dict(giant.degree()).items() if degree < 2]
giant.remove_nodes_from(remove)

bet = nx.centrality.betweenness_centrality(giant)
dc = nx.centrality.degree_centrality(giant)
eig = nx.centrality.eigenvector_centrality(giant)
core = nx.algorithms.core.find_cores(giant)






## one mode projections

authors = bipartite.projected_graph(B,set(edgelist['author']))

assert (len(authors.nodes()) == len(df['author'].unique())), 'number of authors is off'

posts = bipartite.weighted_projected_graph(B, set(edgelist['link_id']))

assert (len(posts.nodes()) == len(df['link_id'].unique())), 'number of posts is off'

nx.connected_components(authors)