# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:15:23 2017

@author: romulo
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
import datetime as dt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot as plot_off
from sklearn.ensemble import RandomForestRegressor



def rmsle(actual, predicted):
    """
    Computes the squared log error.
    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted
    """
    sle = (np.power(np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1), 2))
    msle = np.mean(sle)
    rmsle = np.sqrt(msle)
    return rmsle
    

path = 'C:/Users/romulo/Documents/GitHub/uc_berkeley_ml/project 4/'
t_file = 'train.csv'
test_file = 'test.csv'

def convert_time(x):
    t = dt.datetime.strptime(x,"%m/%d/%Y %H:%M")
    return pd.Series([t.weekday(),t.hour,t.year,t.month,t])


target_columns = ['count','registered','casual']
cat_cols = ['season','holiday','workingday','weekday','hr','yr','month']
regress = ['count']

raw_data = pd.read_csv(path+t_file)
labels = raw_data[regress]
data = raw_data.drop(target_columns,axis=1)
data[['weekday','hr','yr','month','time']]  = data.apply(lambda x: convert_time(x['datetime']),axis=1)


plt = []
weekdays = list(range(1,2))

for d in weekdays:
    line_x = data[data['weekday'] ==1]
    line_y = labels[data['weekday'] ==1]
    plt_x = go.Scatter(x=line_x['hr'],y=line_y['count'],mode = 'lines+markers')
    plt.append(plt_x)
    
#py.plot(plt)
plot_off(plt)


data.drop('datetime',axis=1,inplace = True)
data.drop('time',axis=1,inplace = True)

#print(data.dtypes)
for col in cat_cols:
    data[col] = data[col].astype('category')
#print(data.dtypes)


train_data,test_data = data[:6000],data[6000:]
train_labels, test_labels = labels[:6000], labels[6000:]

#creating an adaBoostRegressor on this
ada = AdaBoostRegressor()
forest = RandomForestRegressor(n_estimators = 1000)

m1 = ada.fit(train_data,train_labels)
f1 = forest.fit(train_data,train_labels)

r = m1.predict(test_data)
f_r = f1.predict(test_data)
print('Accuracy (adaboost with decision trees):', m1.score(test_data, test_labels))
print('Accuracy (random forest):', f1.score(test_data, test_labels))

t_l = np.array(test_labels['count'])
r_l = (np.array(r)).astype('int')
ev =  rmsle(t_l,r_l)
print(ev)

fr_l = (np.array(f_r)).astype('int')
ef =  rmsle(t_l,fr_l)
print(ef)


#compare = test_labels
#compare['predicted'] = r

test_raw_data = pd.read_csv(path+test_file)
test_raw_data [['weekday','hr','yr','month','time']]  = test_raw_data .apply(lambda x: convert_time(x['datetime']),axis=1)
for col in cat_cols:
    test_raw_data[col] = test_raw_data[col].astype('category')
test_raw_data.drop('datetime',axis=1,inplace = True)
test_raw_data.drop('time',axis=1,inplace = True)

 

baseline_r = f1.predict(test_raw_data)
baseline_r_l = (np.array(baseline_r)).astype('int')







