import numpy as np
from numpy import *
from plotBoundary import *
import pylab as pl
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LogisticRegression

# import your LR training code

def sigmoid(x):
    return(1.0/(1.0 + np.exp(-x)))

def dot_and_sigmoid(M, weight):
    M = np.dot(M, weight)
    return sigmoid(M)

# def sgd(X, Y, thresh, learning_rate, reg_factor):
#     dimensions = len(X[0])
#     old_theta = np.array([0] * len(X[0]))
#     new_theta = np.array([-1] * len(X[0]))
#     d = 0
#     while np.linalg.norm(new_theta - old_theta) > thresh:
#         old_theta = copy.deepcopy(new_theta)
#         summation = 0
#         for n in range(len(X)):
#             h = dot_and_sigmoid(X[n], old_theta)
#             first_term = (Y[n][0] - h)*X[n][d]
#             reg_term = 2*reg_factor*math.sqrt(np.dot(old_theta, old_theta))
#             summation += old_theta[d] + learning_rate*(first_term + reg_term)
#         new_theta[d] = summation/len(X)
#         d = (d+1)%dimensions
#         print('new_weights', new_theta)
#         print('old_weights', old_theta)
#         print('threshold', np.linalg.norm(new_theta - old_theta))
#     return new_theta

def sgd(X,Y, thresh, learning_rate, reg_factor, norm):
    samples = len(X)
    old_theta = np.array([0] * len(X[0]))
    new_theta = np.array([0.5] * len(X[0]))
    n = 0
    epoch = 0
    while (epoch < 30):
        old_theta = copy.deepcopy(new_theta)
        second_loss_term = math.exp(-Y[n]*np.dot(old_theta, X[n]))
        first_loss_term = -1.0/(1.0 + second_loss_term)
        third_loss_term = -1*Y[n]*X[n]
        inner_reg_term = X[n]/np.linalg.norm(old_theta[1:])
        reg_term = norm*reg_factor*(inner_reg_term)
        gradient = -first_loss_term*second_loss_term*third_loss_term + reg_term
        change = learning_rate*gradient
        new_theta = old_theta - change
        if n == samples-1:
            epoch +=1
        n = (n+1)%samples
        print(n)
        print(new_theta)
        # if n==4:
        #     asd
    return new_theta

def new_sgd(X,Y, thresh, learning_rate, reg_factor, norm):
    samples = len(X)
    old_theta = np.array([0] * len(X[0]))
    new_theta = np.array([0.5] * len(X[0]))
    n = 0
    epoch = 0
    # while (linalg.norm(old_theta-new_theta) > thresh and epoch <70):
    while(epoch < 50):
        old_theta = copy.deepcopy(new_theta)
        # change for w
        # second_loss_term = math.exp(-Y[n]*np.dot(old_theta[1:], X[n][1:]))
        try:
            second_loss_term = np.exp(-Y[n]*np.dot(old_theta[1:], X[n][1:]))
        except:
            second_loss_term = 0
        first_loss_term = -1.0/(1.0 + second_loss_term)
        third_loss_term = -1*Y[n]*X[n][1:]
        if norm == 2:
            inner_reg_term = 2*old_theta[1:]
        elif norm ==1:
            inner_reg_term = X[n][1:]/np.linalg.norm(old_theta[1:])
        reg_term = reg_factor*(inner_reg_term)
        gradient = -(first_loss_term*second_loss_term*third_loss_term) + reg_term
        # print('first_loss_term', first_loss_term)
        # print('second_loss_term', second_loss_term)
        # print('third_loss_term', third_loss_term)
        # print('reg_term', reg_term)
        # print('gradient',  gradient)

        # change for w0
        # second_loss_term = math.exp(-Y[n]*old_theta[0])
        # second_loss_term = math.exp(-Y[n]*np.dot(old_theta, X[n]))
        try:
            # second_loss_term_w0 = math.exp(-Y[n]*old_theta[0])
            second_loss_term_w0 = np.exp(-Y[n]*np.dot(old_theta, X[n]))
        except:
            second_loss_term_w0 = 0
        first_loss_term_w0 = -1.0/(1.0 + second_loss_term_w0)
        third_loss_term_w0 = -1*Y[n]*old_theta[0]
        gradient_0 = -(first_loss_term_w0*second_loss_term_w0*third_loss_term_w0)

        new_theta = old_theta-learning_rate*(insert(gradient, 0, gradient_0))
        if n == samples-1:
            epoch +=1
        n = (n+1)%samples
    return new_theta



def transform_y(y):
    new_y = []
    for i in range(len(y)):
        new_y.append(y[i][0])
    return new_y

def transform_x(x):
    transformed_X = []
    for i in x:
        transformed_X.append(np.insert(i, 0, 1))
    return transformed_X

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
transformed_X = transform_x(X)
Y = train[:,2:3]

# new_y = []
# for i in range(len(Y)):
#     new_y.append(Y[i][0])
#
# l = LogisticRegression(penalty='l2', C=10000000000000000.0)
# l.fit(X,new_y)
# print(l.score(X, Y))
# print(l.coef_)
# print(l.intercept_)
# def predictLR(x):
#     return l.predict(x)



# Carry out training.
# w = new_sgd(transformed_X, Y, 0.00001, 0.02, 0.01, 2)
# print(w)
# # Define the predictLR(x) function, which uses trained parameters
# def predictLR(x):
#     result = []
#     for sample in x:
#         sample = np.insert(sample, 0, 1)
#         prob = dot_and_sigmoid(sample, w)
#         if prob > 0.5:
#             result.append(1)
#         else:
#             result.append(-1)
#     return np.array(result)
# # #
# # # # plot training results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train with L2 regularization and Lambda = 0')
# pl.show()

# print '======Validation======'
# # load data from csv files
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:,0:2]
# Y = validate[:,2:3]
#
# # plot validation results
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate with L1 regularization and C=10')
#
# print '======Test========'
# #load data from csv files
# test = loadtxt('data/data'+name+'_test.csv')
# X = test[:,0:2]
# Y = test[:,2:3]
#
# plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Test with L1 regularization and C=10')
# pl.show()
# transformed_y = transform_y(Y)
# print('weights', l.coef_, 'w0', l.intercept_)
# print('test_error', l.score(X,transformed_y))

def one_point_two(C, dataset):
    # runs L1 and L2 reg on logistic regression with various C
    datamap = {}
    for name in dataset:
        print('###########-dataset-'+name+'-############')
        train = loadtxt('data/data'+name+'_train.csv')
        test = loadtxt('data/data'+name+'_test.csv')
        validation = loadtxt('data/data'+name+'_validate.csv')
        train_X = train[:,0:2]
        plot_Y = train[:,2:3]
        train_Y = transform_y(plot_Y)
        test_X = test[:,0:2]
        test_Y = test[:,2:3]
        validation_X = validation[:,0:2]
        validation_Y = validation[:,2:3]
        datamap[name] = {}
        for reg in ['l1', 'l2']:
            datamap[name][reg] = []
            print("-----------"+reg+"-------------")
            # fit on different Cs and then score on the test_set
            for c in C:
                print('**'+str(c)+'**')
                l = LogisticRegression(penalty=reg, C=c, intercept_scaling=100)
                l.fit(train_X,train_Y)
                # plotDecisionBoundary(train_X, plot_Y, predictLR, [0.5], title = 'LR Train with '+ reg +' regularization and C=' + str(c))
                # pl.show()
                # print out useful information
                print('weights--->', l.coef_, 'w0--->', l.intercept_)
                print('train_accuracy--->', l.score(train_X, plot_Y))
                print('validation_accuracy--->', l.score(validation_X, validation_Y))
                print('test_accuracy--->', l.score(test_X, test_Y))
                datamap[name][reg].append(l.score(test_X, test_Y))
    return datamap

one_point_two([100.0, 10.0, 1.0, 0.1, 0.5, 0.01], ['1', '2', '3', '4'])
def plots():
    datamap = one_point_two([100.0, 10.0, 1.0, 0.1, 0.5, 0.01], ['1', '2', '3', '4'])

    x = [0,1,2,3,4,5]
    my_ticks = ['0.01', '0.1', '1.0', '2.0', '10.0', '100']
    plt.xticks(x, my_ticks)
    plt.plot(x, datamap['1']['l1'], label='dataset1 L1')
    plt.plot(x, datamap['1']['l2'], label='dataset1 L2')
    plt.plot(x, datamap['2']['l1'], label='dataset2 L1')
    plt.plot(x, datamap['2']['l2'], label='dataset2 L2')
    plt.plot(x, datamap['3']['l1'], label='dataset3 L1')
    plt.plot(x, datamap['3']['l2'], label='dataset3 L2')
    plt.plot(x, datamap['4']['l1'], label='dataset4 L1')
    plt.plot(x, datamap['4']['l2'], label='dataset4 L2')
    plt.xlabel('lambda')
    plt.ylabel('test_accuracy')
    plt.title('Test accuracy with different values of lambda')
    plt.legend(bbox_to_anchor=(0.5, 0.25), loc=0, borderaxespad=0.01, fontsize='small')
    plt.show()
