# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 14:34:54 2017

@author: romulo
"""
# This tells matplotlib not to try opening a new window for each plot.

# Import a bunch of libraries.
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

# Set the randomizer seed so results are the same each time.
np.random.seed(0)

# Load the digit data either from mldata.org, or once downloaded to data_home, from disk. The data is about 53MB so this cell
# should take a while the first time your run it.
mnist = fetch_mldata('MNIST original', data_home='~/datasets/mnist')
X, Y = mnist.data, mnist.target

# Rescale grayscale values to [0,1].
X = X / 255.0

# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

print('data shape: ', X.shape)
print('label shape:', Y.shape)

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[61000:], Y[61000:]
dev_data, dev_labels = X[60000:61000], Y[60000:61000]
train_data, train_labels = X[:60000], Y[:60000]
mini_train_data, mini_train_labels = X[:1000], Y[:1000]

def P1(num_examples=10):
    plt.figure(figsize=(20,20))
    n_digits = 10
    for i in range(n_digits):
        examples = mini_train_data[mini_train_labels==i]
        tbplot = np.random.choice(examples.shape[0],size=num_examples,replace=False)
        for j in range(num_examples):
            plt.subplot(10,n_digits,j+1 +(i*n_digits))
            plt.imshow(examples[tbplot[j],].reshape((28,28)))        


P1()

def P2(k_values):
    for i in k_values:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(mini_train_data,mini_train_labels)
        knn_prediction = knn.predict(dev_data)
        print("Results with K = ", i)
        print("k=", i, "; accuracy:", np.mean(dev_labels == knn_prediction))
        if(i==1):
            print(classification_report(dev_labels,knn_prediction))
    
### STUDENT END ###

k_values = [1, 3, 5, 7, 9]
P2(k_values)

def P3(train_sizes,accuracies):
    for i in train_sizes:
        random_sample = np.random.choice(60000,size=i,replace=False)
        t_data,t_labels = train_data[random_sample,],train_labels[random_sample,]
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(t_data,t_labels)
        t = time.time()
        knn_prediction = knn.predict(dev_data)
        t = time.time() -t
        print("Results with train size = ", i)
        accuracies.append(np.mean(dev_labels == knn_prediction))
        print("Train Size=", i, "; accuracy:", np.mean(dev_labels == knn_prediction))
        print("Seconds to predict: ", t)
        
train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000]
accuracies = []
P3(train_sizes,accuracies)

def P4(predict_size,train_sizes,accuracies):
    lr = LinearRegression()
    t_s = (np.asarray(train_sizes)).reshape((len(accuracies),1))
    t_a = (np.asarray(accuracies)).reshape((len(accuracies),1))
    lr.fit(t_s,t_a)
    predict_accuracy = lr.predict(predict_size)
    return(predict_accuracy)

print(P4(60000,train_sizes,accuracies))

def P5():
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(mini_train_data,mini_train_labels)
    knn_prediction = knn.predict(dev_data)
    return confusion_matrix(dev_labels,knn_prediction,labels = range(10))
    
cf_mtx = P5()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    for i in range(cm.shape[0]):
         for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plot_confusion_matrix(cf_mtx,range(10))


transform = [x**2 for x in train_sizes]
t_s = np.stack((np.asarray(train_sizes),transform),1)

predict_size = 60000
pre_transform = np.asarray([predict_size**2])
pre_t = np.stack((np.asarray([predict_size]),pre_transform),1)


def weightedBlur(pixel,item,size=28):
    '''size = 28
    pixel = 93
    item = test
    '''
    bound = size*size
    weightedVal = []
    #self = 
    wt = 1.0
    totalWeight = wt
    weightedVal.append(item[pixel]*wt)

    #upper = 
    upper_x = pixel-size
    if(upper_x >= 0):
        wt = 0.66
        totalWeight += wt
        weightedVal.append(item[upper_x]*wt)
    #lower = 
    lower_x = pixel+size
    if(lower_x < size*size):
        wt = 0.66
        totalWeight += wt
        weightedVal.append(item[lower_x]*wt)

    #right = 
    right_x = pixel+1
    if((right_x%size > pixel%size) & (right_x < bound)):
        wt = 0.66
        totalWeight += wt
        weightedVal.append(item[right_x]*wt)
        
    #left = 
    left_x = pixel-1
    if(left_x%size < pixel%size):
        wt = 0.66
        totalWeight += wt
        weightedVal.append(item[left_x]*wt)
        
    #upper_right = 
    upper_right_x = pixel-size+1
    if((upper_right_x >= 0) & (upper_right_x%size > pixel%size)):
        wt = 0.33
        totalWeight += wt
        weightedVal.append(item[upper_right_x]*wt)
        
    #lower_right = 
    lower_right_x = pixel+size+1
    if((lower_right_x < bound) & (lower_right_x%size > pixel%size)):
        wt = 0.33
        totalWeight += wt
        weightedVal.append(item[lower_right_x]*wt)
        
    #upper_left = 
    upper_left_x = pixel-size-1
    if((upper_left_x >= 0) & (upper_left_x%size < pixel%size)):
        wt = 0.33
        totalWeight += wt
        weightedVal.append(item[upper_left_x]*wt)
        
    #lower_left = 
    lower_left_x = pixel+size-1
    if((lower_left_x < bound) & (lower_left_x%size < pixel%size)):
        wt = 0.33
        totalWeight += wt
        weightedVal.append(item[lower_left_x]*wt)
    
    val = sum(weightedVal)/totalWeight
    return val
        
    
def plot_num(item):   
    plt.figure(figsize=(20,20))
    plt.subplot(10,10,1)
    plt.imshow(item.reshape((28,28)))
    
def blur(item):
    blur = []
    for x in range(item.shape[0]):
        blur.append(weightedBlur(x,item))
        
    return np.asarray(blur)


test = mini_train_data[25,]

test_blur = blur(test)

blur_mini_train_data = np.apply_along_axis(blur,1,mini_train_data)
#test_blur = fv(test,test)

plot_num(test)
plot_num(test_blur)

plt.figure(figsize=(20,20))
plt.subplot(10,10,1)
plt.imshow(test.reshape((28,28)))


nb = BernoulliNB(binarize = 0.4)
nb.fit(mini_train_data,mini_train_labels)
proa = nb.predict_proba(dev_data)
m = np.apply_along_axis(np.argmax,1,proa)
mm = np.apply_along_axis(max,1,proa)
mmm = np.asarray([find_nearest(buckets,x) for x in mm])

correct = (m == (dev_labels))
co_e = (correct[(mmm==0.999)])

count = (mmm == 0.999).sum()
sub = m[(mmm==0.999)] 

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


buckets = [0.5, 0.9, 0.999, 0.99999, 0.9999999, 0.999999999, 0.99999999999, 0.9999999999999, 1.0]
correct = [0 for i in buckets]
total = [0 for i in buckets]

nb = BernoulliNB(binarize = 0.4)
nb.fit(mini_train_data,mini_train_labels)
proba = nb.predict_proba(dev_data)
prediction = np.apply_along_axis(np.argmax,1,proba)
max_prob = np.apply_along_axis(max,1,proba)
bucket = np.asarray([find_nearest(buckets,x) for x in max_prob])
for i in range(len(buckets)):
    total[i] = (bucket==(buckets[i])).sum()
    correct = (prediction == dev_labels)
    correct[i] = (correct[(bucket==(buckets[i]))]).sum()



bucket = find_nearest(buckets,proa[0,m])

p = nb.feature_log_prob_
prob = np.exp(p)
ran = np.random.rand(10,784)
resu = (ran<=prob)*ran
plot_num(resu[0,])


def wgb_class(data,t):
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if(data[x,y]<= t[0]):
                data[x,y] = 0
            else:
                if(data[x,y]<=t[1]):
                    data[x,y]= 1
                else:
                    data[x,y]= 2
    return data

wgb_test = wgb_class(test,[0.33,0.66,1.00])

alphas = {'alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]}

def Gaussian_Test():
    #sample_wt = np.random.RandomState(42).rand(mini_train_labels.shape[0])
    nb = GaussianNB()
    nb.fit(mini_train_data,mini_train_labels)
    nb_prediction = nb.predict(dev_data)
    print("Accuracy:", np.mean(dev_labels == nb_prediction))
    return nb

gaussian = Gaussian_Test()

gaussian.sigma_ 

gaussian.
feature_log_prob_


