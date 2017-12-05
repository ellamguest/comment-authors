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
df['post_id'] = df.link_id.str.lstrip('t3_')
df['time'] = pd.to_datetime(df['created_utc'], unit='s')

df = df[~df['author'].isin(['[deleted]','AutoModerator', 'DeltaBot'])]

edgelist = df[['author','post_id']]

B = nx.Graph()
B.add_nodes_from(set(edgelist['author']), bipartite=0) 
B.add_nodes_from(set(edgelist['post_id']), bipartite=1)
B.add_edges_from(list(zip(edgelist['author'],edgelist['post_id'])))

assert (len(B.nodes()) == len(df['author'].unique()) + len(df['post_id'].unique())), 'number of nodes is off'
assert (len(B.edges()) == len(edgelist.drop_duplicates())), 'number of edges is off'
print(f'the two-mode network has {len(B.nodes())} nodes'
      f' and density of {nx.density(B)}')

## one mode projections
authors = bipartite.projected_graph(B,set(edgelist['author']))
assert (len(authors.nodes()) == len(df['author'].unique())), 'number of authors is off'
print(f'the author network has {len(authors.nodes())} nodes'
      f' and density of {nx.density(authors)}')

posts = bipartite.weighted_projected_graph(B, set(edgelist['post_id']))
assert (len(posts.nodes()) == len(df['post_id'].unique())), 'number of posts is off'
print(f'the post network has {len(posts.nodes())} nodes'
      f' and density of {nx.density(posts)}')

# mods
date = '2017-10-27'
sub = 'cmv'
nodelist = pd.read_csv('/Users/emg/Programming/GitHub/mod-timelines/moding-data/{}/{}/lists/nodelist.csv'.format(sub,date))

mods = set(nodelist[nodelist['type']==1]['name'])

def get_giant(net):
    subgraphs = list(nx.connected_component_subgraphs(net))
    giant = sorted(subgraphs, key=len)[-1]
    return giant

def get_table(net):
    print('getting coreness...')
    core = nx.find_cores(net)  
    print('getting degrees...')
    deg = dict(nx.degree(net))
    print('getting degree centrality...')
    dc = nx.degree_centrality(net)
    print('getting eigenvector centrality...')
    eig = nx.eigenvector_centrality_numpy(net)
    print('compiling table...')
    table = pd.DataFrame({'degree':deg, 'degree_cent':dc,
                  'coreness':core,
                  'eigenvector':eig},
                index=list(net.nodes()))
    return table

def scatter_table(table, title='author centrality, colour = coreness'):
    '''w/ columns degree, eigenvector, coreness'''
    x, y, c = 'degree', 'eigenvector', 'coreness'
    plt.scatter(x = table[x], y = table[y], c = table[c])
    plt.xlabel('degree centrality')
    plt.ylabel('eigenvector centrality')
    plt.title(title)

author_giant = get_giant(authors)
author_table = get_table(authors)

mod_table = author_table.loc[mods].fillna(False)
missing = mod_table[mod_table['degree']==False]
present = mod_table[mod_table['degree']!=False]

print(f'there are {len(missing)} mods who did not make comments that month')    

scatter_table(author_table)
scatter_table(present)

post_giant = get_giant(posts)
post_table = get_table(post_giant)

scatter_table(post_table)

plt.hist(list(dict(nx.degree(posts)).values()))
plt.hist(list(dict(nx.degree_centrality(posts)).values()))
plt.hist(list(dict(nx.find_cores(posts)).values()))
plt.hist(post_table['coreness'])
plt.hist(list(dict(nx.eigenvector_centrality_numpy(posts)).values()))
plt.hist(post_table['eigenvector'])

plt.hist(list(dict(nx.degree(authors)).values()))
plt.hist(list(dict(nx.degree_centrality(authors)).values()))
plt.hist(list(dict(nx.find_cores(authors)).values()))
plt.hist(author_table['coreness'])
plt.hist(list(dict(nx.eigenvector_centrality_numpy(authors)).values()))
plt.hist(author_table['eigenvector'])


# post data
post_data = pd.read_pickle('/Users/emg/Programming/GitHub/comment-authors/cmv_17_06_posts.pkl')
post_data['time'] = pd.to_datetime(post_data['created_utc'], unit='s')

post_titles = dict(zip(post_data['id'], post_data['title']))
post_times = dict(zip(post_data['id'], post_data['time']))

print('MOST CENTRAL POSTS')
print()
for post in post_table.sort_values('degree').tail().index:
    if post in post_titles.keys():
        print(post_times[post])
        print(post_titles[post])
    else:
        print('post not found')
    print()

print('LEAST CENTRAL POSTS')
print()
for post in post_table.sort_values('degree').head(10).index:
    if post in post_titles.keys():
        print(post_times[post])
        print(post_titles[post])
    else:
        print('post not found')
    print()

post_table['time'] = post_table.index.map(lambda x: post_times.get(x)).fillna(False)
missing_posts = post_table[post_table['time']==False]

print(f'there are {len(missing_posts)} missing posts out of {len(post_table)}'
      f' or {len(missing_posts)/len(post_table)}')


post_utc = dict(zip(post_data['id'], post_data['created_utc']))
post_table['utc'] = post_table.index.map(lambda x: post_utc.get(x)).fillna(False)
present_posts = post_table[post_table['time']!=False]

plt.scatter(present_posts['utc'], present_posts['coreness'])  
plt.xlabel('time')
plt.ylabel('coreness')
plt.title('post giant coreness by time')


scatter_table(post_table)
scatter_table(missing_posts)
scatter_table(present_posts)


B_table = get_table(B)

x = list(dict(nx.degree_centrality(posts)).values())
y = list(dict(nx.find_cores(posts)).values())
plt.scatter(x,y)
plt.xlabel('degree centrality')
plt.ylabel('coreness')
plt.title('post giant coreness by degree centrality')

#### removing mod ties from network

modless = df[~df['author'].isin(mods)]
modposts = df[df['author'].isin(mods)]

def nets(df):
    edgelist = df[['author','post_id']]
    
    B = nx.Graph()
    B.add_nodes_from(set(edgelist['author']), bipartite=0) 
    B.add_nodes_from(set(edgelist['post_id']), bipartite=1)
    B.add_edges_from(list(zip(edgelist['author'],edgelist['post_id'])))
    
    assert (len(B.nodes()) == len(df['author'].unique()) + len(df['post_id'].unique())), 'number of nodes is off'
    assert (len(B.edges()) == len(edgelist.drop_duplicates())), 'number of edges is off'
    print(f'the two-mode network has {len(B.nodes())} nodes'
          f' and density of {nx.density(B)}')
    
    ## one mode projections
    authors = bipartite.projected_graph(B,set(edgelist['author']))
    assert (len(authors.nodes()) == len(df['author'].unique())), 'number of authors is off'
    print(f'the author network has {len(authors.nodes())} nodes'
          f' and density of {nx.density(authors)}')
    
    posts = bipartite.weighted_projected_graph(B, set(edgelist['post_id']))
    assert (len(posts.nodes()) == len(df['post_id'].unique())), 'number of posts is off'
    print(f'the post network has {len(posts.nodes())} nodes'
          f' and density of {nx.density(posts)}')

    return B, authors, posts

mB, Mauthors, Mposts = nets(modposts)
lB, lauthors, lposts = nets(modless)

Mpost_table = get_table(Mposts)

lpost_table = get_table(lposts)
scatter_table(Mpost_table)
scatter_table(post_table)

scatter_table(lpost_table)

t = get_table(lauthors)
scatter_table(t)

x = list(dict(nx.degree_centrality(Mposts)).values())
y = list(dict(nx.find_cores(Mposts)).values())
plt.scatter(x,y)
plt.xlabel('degree centrality')
plt.ylabel('coreness')
plt.title('mod post giant coreness by degree centrality')

print('MOST CORE MOD POSTS')
print()
for post in Mpost_table.sort_values('coreness').tail().index:
    if post in post_titles.keys():
        print(post_times[post])
        print(post_titles[post])
    else:
        print('post not found')
    print()

pos=nx.spring_layout(Mauthors)
nx.draw(Mauthors, pos=pos)   
nx.draw_networkx_labels(Mauthors, pos=pos)

mod_author_table = get_table(Mauthors)
modless_author_table = get_table(lauthors)

B_giant = get_giant(posts)
c = list(nx.clique.find_cliques(B_giant))
c = sorted(c, key=len)



#### ANDY SVD
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import scipy as sp

X = nx.to_numpy_matrix(B)


X = c.as_matrix()
N = Normalizer().fit_transform(X)
U, D, Vh = sp.linalg.svd(N)


plt.scatter(*Vh[:, :2].T)
plt.scatter(*U[:, :2].T)
plt.scatter(*U[:2])
plt.legend()

edgelist['count']=1
aff = edgelist.pivot_table(index='author',columns='post_id', values='count')

aff = aff.loc[(aff.sum(axis=1) > 4), (aff.sum(axis=0) > 4)]

plt.scatter(
    data[:, 0], data[:, 1], marker='o', c=data[:, 2], s=data[:, 3] * 1500,
    cmap=plt.get_cmap('Spectral'))


x = U[:, 0]
y = U[:, 1]
labels = c.index

plt.scatter(x,y)
for i, label in enumerate(labels):
    plt.annotate(label,
        xy=(x[i], y[i]), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
