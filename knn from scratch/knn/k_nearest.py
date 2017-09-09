# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:12:37 2017

@author: romulo
"""
from collections import Counter
from knn import distance

class KNearestNeighbors:
    # Initialize an instance of the class.
    def __init__(self, metric=distance.EuclideanDistance, k=1):
        self.metric = metric
        self.k = k 
    
    # No training for Nearest Neighbors. Just store the data.
    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
    
    # Make predictions for each test example and return results.
        
    def predict(self, test_data):
        results = list(map(self._predict_item,test_data))
        """for item in test_data:
            results.append(self._predict_item(item))
        """
        return results
    
        
    def _predict_item(self,item):
        ## implement this !
        neighbors = []
        for i in range(len(self.train_data)):
            dist = self.metric(self.train_data[i], item)
            neighbors += [(dist,self.train_labels[i])]
        neighbors.sort(key = lambda x: x[0])
        kNN = neighbors[0:self.k]
        labels = Counter(label[1] for label in kNN)
        best_label = labels.most_common(1)[0][0]
        return best_label