# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:09:56 2017

@author: romulo
"""


# General libraries.
import re
import numpy as np
import matplotlib.pyplot as plt

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

# SK-learn library for importing the newsgroup data.
from sklearn.datasets import fetch_20newsgroups

# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *


categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)

num_test = len(newsgroups_test.target)
split = int(num_test/2)
test_data, test_labels = newsgroups_test.data[split:], newsgroups_test.target[split:]
dev_data, dev_labels = newsgroups_test.data[:split], newsgroups_test.target[:split]
train_data, train_labels = newsgroups_train.data, newsgroups_train.target

print('training label shape:', train_labels.shape)
print('test label shape:', test_labels.shape)
print('dev label shape:', dev_labels.shape)
print('labels names:', newsgroups_train.target_names)


for i in range(5):
    print("Example", i+1)
    print("Label", newsgroups_train.target_names[train_labels[i]])
    print("Msg:", train_data[i])


#countvectorizer
np.random.seed(0)

features = []
c_vec = [0.001,0.01, 0.1, 1.0,10,100]
for c in c_vec:
    features = features + [c]

plt.plot(np.log(c_vec), np.array(features))
plt.title('L2 Accuracy based on C used in L1')
plt.ylabel('Accuracy') 
plt.xlabel('log(C)') 
plt.show()

C = 100
np.random.seed(0)
vectorizer = TfidfVectorizer()
vector_train_data = vectorizer.fit_transform(train_data)
vector_dev_data = vectorizer.transform(dev_data)
lr = LogisticRegression(C=C,penalty="l2")
lr.fit(vector_train_data,train_labels)
resu = lr.predict(vector_dev_data)
print("F1 : ", metrics.f1_score(dev_labels,resu,average = 'weighted'))
acc = metrics.accuracy_score(dev_labels,resu)
print("Accuracy: ",acc)

proba = lr.predict_proba(vector_dev_data)

R = np.zeros(dev_labels.shape)
for z in range(dev_labels.shape[0]):
    prob_cat = proba[z][dev_labels[z]]
    max_prob = np.max(proba[z])
    R[z] = max_prob/prob_cat
    print(prob_cat,max_prob)

idx = R.argsort()[-3:][::-1]
for i in range(len(idx)):
    print(train_data[idx[i]:])
