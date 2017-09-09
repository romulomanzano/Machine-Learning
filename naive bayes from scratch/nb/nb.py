# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:06:16 2017

@author: romulo
"""
import numpy as np

class NaiveBayes:
    # Initialize an instance of the class.
    def __init__(self, alpha=0.0):
        self.alpha = alpha     # additive (Laplace) smoothing parameter
        self.priors = None     # estimated by fit()
        self.probs = None      # estimated by fit()
        self.num_labels = 0    # set by fit()
        self.num_features = 0  # set by fit()
        
    def fit(self, train_data, train_labels):
        # Store number of labels, number of features, and number training examples.
        self.num_labels = len(np.unique(train_labels))
        self.num_features = train_data.shape[1]
        self.num_examples = train_data.shape[0]
        
        # Initialize an array of label counts. Each label gets a smoothed count of 2*alpha because
        # each feature value (0 and 1) gets an extra count of alpha.
        label_counts = np.ones(self.num_labels) * self.alpha * 2

        # Initialize an array of (feature=1, label) counts to alpha.
        feature0_and_label_counts = np.ones([self.num_features, self.num_labels]) * self.alpha
    
        # Count features with value == 1.
        for i in range(self.num_examples):
            label = train_labels[i]
            label_counts[label] += 1
            for feature_index, feature_value in enumerate(train_data[i]):
                feature0_and_label_counts[feature_index][label] += (feature_value == 1)

        # Normalize to get probabilities P(feature=1|label).
        self.probs = feature0_and_label_counts / label_counts
        
        # Normalize label counts to get prior probabilities P(label).
        self.priors = label_counts / label_counts.sum()

    # Make predictions for each test example and return results.
    def predict(self, test_data):
        results = []
        for item in test_data:
            results.append(self._predict_item(item))
        return np.array(results)
    
    # Private function for making a single prediction.
    def _predict_item(self, item):
        # Make a copy of the prior probabilities.
        predictions = self.priors.copy()
        
        # Multiply by each conditional feature probability.
        for (index, value) in enumerate(item):
            feature_probs = self.probs[index]
            if not value: feature_probs = 1 - feature_probs
            predictions *= feature_probs

        # Normalize and return the label that gives the largest probability.
        predictions /= predictions.sum()
        return predictions.argmax()