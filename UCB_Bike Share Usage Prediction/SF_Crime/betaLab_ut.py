# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:16:14 2017

@author: romulo
"""

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

#Basic variables defined to load the data and support the analysis
path = 'C:/Users/romulo/Dropbox/08_Kaggle/SF Crime/Data/'
t_file = 'train.csv'
test_file = 'test.csv'

#quick function needed to breakout the timestamp
def convert_time(x):
    t = dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
    return pd.Series([t.weekday(),t.hour,t.year,t.month,t])

#loading 
raw_data = pd.read_csv(path+t_file)
#features vs target
#Categorical cols
cat_features = ['Resolution','DayOfWeek','PdDistrict']
labels = raw_data['Category']
data = raw_data.drop(['Category'],axis=1)
#Get dummy data
dummy_data = pd.concat([data,pd.get_dummies(data[cat_features])],axis=1)
dummy_data = dummy_data.drop(cat_features,axis = 1)
#further drop additional columns as not that much time for feature engineering
dummy_data = dummy_data.drop(['Descript','Address'],axis = 1)
#Split Time into separate columns
dummy_data[['weekday','hr','yr','month','time']]  = dummy_data.apply(lambda x: convert_time(x['Dates']),axis=1)
dummy_data = dummy_data.drop(['Dates','time'],axis = 1)


#CONTINUE HERE!!! Need to break:
    ##Streets into viable elements
    ##Cluster XY into neighborhoods
    ## Assign clusters a number
    ##Block vs. Intersection
    ## Get all streets and harmonize (UPPERCASE and Dummy out)
    ## Get hours, month, year
    
## test vs. training data split
train_features, test_features, train_labels, test_labels = train_test_split(dummy_data,labels,test_size = 0.3,random_state = 0)


#train_data,test_data = raw_data[:6000],raw_data[6000:]
#train_labels, test_labels = raw_labels[:6000], raw_labels[6000:]
#Fit models


tr = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
tr.fit(train_features, train_labels)

print('Accuracy (a decision tree):', tr.score(test_features, test_labels))

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_features, train_labels)

print('Accuracy (a random forest):', rfc.score(test_features, test_labels))

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.1)

abc.fit(train_features, train_labels)
print('Accuracy (adaboost with decision trees):', abc.score(test_features, test_labels))



