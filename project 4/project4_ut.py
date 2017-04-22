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
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor



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
    


## Actual problem training starts here
path = 'C:/Users/romulo/Documents/GitHub/uc_berkeley_ml/project 4/'
t_file = 'train.csv'
test_file = 'test.csv'

def convert_time(x):
    t = dt.datetime.strptime(x,"%m/%d/%Y %H:%M")
    return pd.Series([t.weekday(),t.hour,t.year,t.month])


target_columns = ['count','registered','casual']
regress = ['count']
regress_registered = ['registered']
regress_casual = ['casual']

raw_data = pd.read_csv(path+t_file)
labels = raw_data[regress]
labels_registered = raw_data[regress_registered]
labels_casual = raw_data[regress_casual]

data = raw_data.drop(target_columns,axis=1)
data[['weekday','hr','yr','month']]  = data.apply(lambda x: convert_time(x['datetime']),axis=1)


cat_cols = ['season','holiday','workingday','weekday','hr','yr','month']
binarize_cols = ['season','weather']#,'yr']

data.drop('datetime',axis=1,inplace = True)
#data.drop('time',axis=1,inplace = True)

## binarizing?
data = pd.get_dummies(data,columns = binarize_cols )

#used 33
train_data,test_data,train_labels ,test_labels = train_test_split(data, labels, test_size=0.20, random_state=42)

t_l = np.array(test_labels['count'])

#creating an adaBoostRegressor on this
forest = RandomForestRegressor(n_estimators = 1000,n_jobs = -1)
extraTree = ExtraTreesRegressor(n_estimators = 1000,n_jobs = -1)

forest_reg = RandomForestRegressor(n_estimators = 1000,n_jobs = -1)
forest_cas = RandomForestRegressor(n_estimators = 1000,n_jobs = -1)


f1 = forest.fit(train_data,train_labels)
f_r = f1.predict(test_data)
fr_l = (np.array(f_r)).astype('int')
ef =  rmsle(t_l,fr_l)
print('Accuracy (random forest):', f1.score(test_data, test_labels))
print('RMSLE with random forest:', ef)


#Extra trees
ex1 = extraTree.fit(train_data,train_labels)
ex_r = ex1.predict(test_data)
ex_l = (np.array(ex_r)).astype('int')
ex =  rmsle(t_l,ex_l)
print('Accuracy (extra Trees):', ex1.score(test_data, test_labels))
print('RMSLE with extratrees:', ex)



#composite:
'''    
train_labels_registered, test_labels_registered = labels_registered[:6000], labels_registered[6000:]
train_labels_casual, test_labels_casual = labels_casual[:6000], labels_casual[6000:]
'''


data_comp = raw_data.drop(regress,axis=1)
labels_comp = raw_data[regress]
data_comp[['weekday','hr','yr','month']]  = data_comp.apply(lambda x: convert_time(x['datetime']),axis=1)


cat_cols = ['season','holiday','workingday','weekday','hr','yr','month']


data_comp.drop('datetime',axis=1,inplace = True)
#data_comp.drop('time',axis=1,inplace = True)

#print(data.dtypes)
for col in cat_cols:
    data_comp[col] = data_comp[col].astype('category')
#print(data.dtypes)

## binarizing?
data_comp = pd.get_dummies(data_comp,columns = binarize_cols)

#started with 33
train_data_comp,test_data_comp,train_labels_comp,test_labels_comp = train_test_split(data_comp, labels_comp, test_size=0.20, random_state=42)

train_labels_reg = train_data_comp[regress_registered]
train_labels_cas = train_data_comp[regress_casual]

train_data_comp.drop(regress_registered,axis=1,inplace=True)
train_data_comp.drop(regress_casual,axis=1,inplace=True)
test_data_comp.drop(regress_registered,axis=1,inplace=True)
test_data_comp.drop(regress_casual,axis=1,inplace=True)




f1_registered = forest_reg.fit(train_data_comp,train_labels_reg)
f_r_registered = f1_registered.predict(test_data_comp)


f1_casual = forest_cas.fit(train_data_comp,train_labels_cas)
f_r_casual = f1_casual.predict(test_data_comp)

f_r_composite = f_r_registered + f_r_casual
f_r_composite_l = (np.array(f_r_composite)).astype('int')
ef_composite =  rmsle(t_l,f_r_composite_l)
print('RMSLE with random forest composite:', ef_composite)


#averages
f_r_avg = (f_r_composite_l + fr_l )/2
f_r_avg_l = (np.array(f_r_avg)).astype('int')
ef_avg =  rmsle(t_l,f_r_avg_l)
print('RMSLE with random forest composite and full avg:', ef_avg)



### Prepare for submission

submit_raw_data = pd.read_csv(path+test_file)
submit_raw_data [['weekday','hr','yr','month','time']]  = submit_raw_data .apply(lambda x: convert_time(x['datetime']),axis=1)
for col in cat_cols:
    submit_raw_data[col] = submit_raw_data[col].astype('category')
    
submit_result = (submit_raw_data['datetime']).to_frame()
submit_result_full = (submit_raw_data['datetime']).to_frame()

submit_raw_data.drop('datetime',axis=1,inplace = True)
submit_raw_data.drop('time',axis=1,inplace = True)

submit_raw_data = pd.get_dummies(submit_raw_data,columns = ['season','weather','yr'])


submit_registered = f1_registered.predict(submit_raw_data)
submit_casual = f1_casual.predict(submit_raw_data)

submit_composite = submit_registered + submit_casual
submit_composite_l = (np.array(submit_composite)).astype('int')

submit_result['count'] = submit_composite_l

#regular one
submit_full = f1.predict(submit_raw_data)
submit_full_l = (np.array(submit_full)).astype('int')
submit_result_full['count'] = submit_full_l

