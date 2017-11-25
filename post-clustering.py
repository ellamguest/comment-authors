#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:38:51 2017

@author: emg
"""

import pandas as pd
import textacy # code for textacy version 0.3.4, update when 0.4 in conda forge
import networkx as nx
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('cmv_17_06_posts.pkl')

print('there are {} posts'.format(df.shape[0]))
print('there are {} authors'.format(len(df.author.unique())))

subset = df[~df['author'].isin(['[deleted]','AutoModerator'])]
has_body = subset[~subset['selftext'].isin(['[deleted]','[removed]', '[]'])]

titles = list(df['title'])

#has_body['comments'] = has_body.num_comments.apply(lambda x: 1 if x > 0 else 0)
has_body = has_body[has_body['num_comments']>0]
has_body = has_body.reset_index(drop=True)
bodies = list(has_body['selftext'])

clean_bodies = [textacy.preprocess.preprocess_text(text, fix_unicode=True,
                                                   no_urls=True, no_emails=True,
                                                   no_phone_numbers=True, no_numbers=True,
                                                   no_currency_symbols=True, no_punct=True,
                                                   no_contractions=True, no_accents=True)
                for text in bodies]

corpus = textacy.Corpus('en', texts=clean_bodies)
corpus

doc_term_matrix, id2term = textacy.vsm.doc_term_matrix(
        (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
        for doc in corpus),
        weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)

print(repr(doc_term_matrix))


model = textacy.tm.TopicModel('nmf', n_topics=10)
model.fit(doc_term_matrix)
doc_topic_matrix = model.transform(doc_term_matrix)
doc_topic_matrix.shape

topics = []
for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    topics.append(top_terms[0])
    print('topic', topic_idx, ':', ' '.join(top_terms))

topic_weights = pd.Series(model.topic_weights(doc_topic_matrix), index=topics)

n = nx.from_numpy_array(doc_topic_matrix.T.dot(doc_topic_matrix))

label_dict = dict(enumerate(topics))
weights=[d['weight']*2 for u,v,d in n.edges(data=True)]

nx.draw(n, with_labels=True, width=weights, labels=label_dict)

d = nx.from_numpy_array(doc_topic_matrix.dot(doc_topic_matrix.T))
nx.draw(d, with_labels=True)

# 0 = hub
# 57 = isolate

has_body['log_num_comments'] = np.log(has_body['num_comments'])
topic_matrix = pd.DataFrame(doc_topic_matrix)

topic_matrix['log_num_comments'] = has_body['log_num_comments']

train_cols = topic_matrix.columns[:-1]

logit = sm.GLM(topic_matrix['log_num_comments'], topic_matrix[train_cols])
result = logit.fit()

.reset_index(drop=True)
has_body = has_body.reset_index(drop=True)
x = has_body[has_body['num_comments']>0]

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

train = topic_matrix[:-140]
test = topic_matrix[-140:]

train_x, train_y = train[train_cols], train['log_num_comments']

test_x, test_y = test[train_cols], test['log_num_comments']

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the testing set
y_pred = regr.predict(test_x)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, y_pred))

#plt.scatter(test_x, test_y,  color='black')
plt.plot(test_x, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


