# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:05:46 2017

@author: romulo
"""

import numpy as np
from sklearn.datasets import load_iris
import knn
#from knn import distance

iris = load_iris()
print('Iris target names:', iris.target_names)
print('Iris feature names:', iris.feature_names)
X, Y = iris.data, iris.target

# Shuffle the data, but make sure that the features and accompanying labels stay in sync.
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

# Split into train and test.
train_data, train_labels = X[:100], Y[:100]
test_data, test_labels = X[100:], Y[100:]

print(X.shape)

        
## code goes here!
for k in range(1, 10):
    clfk = knn.k_nearest.KNearestNeighbors(k=k,metric = knn.distance.LInfinityDistance)
    clfk.fit(train_data, train_labels)
    preds = clfk.predict(test_data)
    
    print("k=", k, "; accuracy:", np.mean(preds == test_labels))        
