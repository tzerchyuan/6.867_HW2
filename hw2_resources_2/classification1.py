from numpy import *
from lr_test import *
# import data
train_1 = loadtxt('data/mnist_digit_1.csv')
train_7 = loadtxt('data/mnist_digit_7.csv')
X_1 = train_1[:500]
X_7 = train_7[:500]
train_X = X_1[:200] + X_7[:200]
train_Y = [1]*200 + [-1]*200

validate_X = X_1[200:350] + X_7[200:350]
validate_Y = [1]*150 + [-1]*150

test_X = X_1[350:500] + X_7[350:500]
test_Y = [1]*150 + [-1]*150

# create corresponding sets with normalize pixels
def normalize(X):
    # returns normalized datasets
    new_X = copy.deepcopy(X)
    return new_X*2/255-1

normalized_train_X = normalize(train_X)
normalized_validate_X = normalize(validate_X)
normalized_test_X = normalize(test_X)

def logistic_classify():
    # trains on training set and selects hyperparameter on validation set
    # returns score on test_set
    return
