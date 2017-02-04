# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 20:08:51 2017

@author: romulo
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 

from sklearn.datasets import make_gaussian_quantiles

#use some simulated data: concentric spheres of classes, see plots and examples here:
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

X, Y = make_gaussian_quantiles(cov=2.,
                                 n_samples=4000, n_features=10,
                                 n_classes=2, random_state=1)

np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

train_data, train_labels = X[:2000], Y[:2000]
test_data, test_labels = X[2000:], Y[2000:]

#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html


dt = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
dt.fit(train_data, train_labels)

print('Accuracy (a decision tree):', dt.score(test_data, test_labels))

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(train_data, train_labels)

print('Accuracy (a random forest):', rfc.score(test_data, test_labels))

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000, learning_rate=0.1)

abc.fit(train_data, train_labels)
print('Accuracy (adaboost with decision trees):', abc.score(test_data, test_labels))


###Own implementation of Bagging:

np.random.seed(1)

B = 1000
n = train_data.shape[0]
sn = int(n*2.0/3.0)   # nr of training data in subset for each tree
nf = train_data.shape[1]
all_preds = np.zeros((B,test_data.shape[0]))

for b in range(B):
    bs_sample_index = np.random.choice(range(n), size=sn, replace=True)
    subt = train_data[bs_sample_index,]
    sublabl = train_labels[bs_sample_index,]
    dt.fit(subt, sublabl)
    sco = dt.predict(test_data)
    all_preds[b,] = sco

voting = np.sum(all_preds,axis=0) / B
voting = [int(x >= 0.5) for x in voting]
print('Accuracy of bagging implementation:', np.mean(voting==test_labels))

###Own implementation of Random Forest:

np.random.seed(1)

B = 1000
n = train_data.shape[0]
sn = int(n*2.0/3.0)   # nr of training data in subset for each tree
nf = train_data.shape[1]
all_preds = np.zeros((B,test_data.shape[0]))

for b in range(B):
    bs_sample_index = np.random.choice(range(n), size=sn, replace=True)
    feature_index = np.random.choice(range(nf), size=int(nf**(1.0/2)), replace=False)
    subt = (train_data[bs_sample_index,:])[:,feature_index]
    sublabl = train_labels[bs_sample_index,]
    dt.fit(subt, sublabl)
    sco = dt.predict(test_data[:,feature_index])
    all_preds[b,] = sco

voting = np.sum(all_preds,axis=0) / B
voting = [int(x >= 0.5) for x in voting]
print('Accuracy random forest implementation:', np.mean(voting==test_labels))