# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:10:52 2017

@author: romulo
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import BernoulliNB
import nb

iris = load_iris()
X,Y = iris.data,iris.target

#rearranging
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X,Y = X[shuffle], Y[shuffle]
#spliting into training and test data
train_data, train_labels = X[:100], Y[:100]
test_data, test_labels = X[100:], Y[100:]

def binarize_iris(data, threshold = [6.0,3.0,2.5,1.0]):
    binarized = np.zeros(data.shape)
    for feature in range(data.shape[1]):
        binarized[:,feature] = data[:,feature] > threshold[feature]
    return binarized

binarized_train_data = binarize_iris(train_data)
binarized_test_data = binarize_iris(test_data)


alpha = 1
nby = nb.nb.NaiveBayes(alpha=alpha)
nby.fit(binarized_train_data, train_labels)

# Compute accuracy on the test data.
preds = nby.predict(binarized_test_data)
correct, total = 0, 0
for pred, label in zip(preds, test_labels):
    if pred == label: correct += 1
    total += 1
print('With alpha = %.2f' %alpha)
print('[OUR implementation] total: %3d  correct: %3d  accuracy: %3.2f' %(total, correct, 1.0*correct/total))




# Compare to sklearn's implementation.
clf = BernoulliNB(alpha=alpha, fit_prior=False)
clf.fit(binarized_train_data, train_labels)
print('sklearn accuracy: %3.2f' %clf.score(binarized_test_data, test_labels))

print('\nOur feature probabilities\n', nby.probs)
print('\nsklearn feature probabilities\n', np.exp(clf.feature_log_prob_).T)
print('\nOur prior probabilities\n', nby.priors)
print('\nsklearn prior probabilities\n', np.exp(clf.class_log_prior_))
