from numpy import *
# from plotBoundary import *
# import pylab as pl
import matplotlib.pyplot as plt
from Problem21 import run_SVM_kernel, linear_kernel, RBF_kernel
from lr_test import *
# import data
train_4 = loadtxt('data/mnist_digit_4.csv')
train_9 = loadtxt('data/mnist_digit_9.csv')
X_4 = train_4[:500]
X_9 = train_9[:500]

train_X = concatenate((X_4[:200], X_9[:200]))
print(train_X[:10,:])
train_Y = [1]*200 + [-1]*200
print(len(train_Y))
validate_X = concatenate((X_4[200:350],X_9[200:350]))
validate_Y = [1]*150 + [-1]*150

test_X = concatenate((X_4[350:500], X_9[350:500]))
test_Y = [1]*150 + [-1]*150

# create corresponding sets with normalize pixels
def normalize(X):
    # returns normalized datasets
    new_X = copy.deepcopy(X)
    return new_X*2/255-1

normalized_train_X = normalize(train_X)
normalized_validate_X = normalize(validate_X)
normalized_test_X = normalize(test_X)
