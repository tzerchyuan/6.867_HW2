from numpy import *
from plotBoundary import *
import pylab as pl
import random as rand
# import your LR training code


# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

print(X.shape)
# Carry out training.
N = Y.shape[0]


def run_Pegasos(X, Y, L, max_epochs):
    d = X.shape[1]
    t = 0
    w = zeros(d)
    w0 = 0
    epoch = 0

    while epoch < max_epochs:
        for i in rand.sample(range(N), N):
            t += 1
            eta = 1/(t*L)
            if Y[i,0]*(dot(w, X[i,:]) + w0) < 1:
                w = (1 - eta * L) * w + eta*Y[i,0]*X[i,:]
                w0 += eta * Y[i,0]
            else:
                w = (1 - eta * L) * w
            # print(w0, w)
        epoch += 1

    return w0, w

L = .02
max_epochs = 100
# w0, w = run_Pegasos(X, Y, L, max_epochs)

def run_kernelized_Pegasos(X, Y, L, max_epochs, M):
    t = 0
    a = zeros(N)
    epoch = 0

    while epoch < max_epochs:
        for i in rand.sample(range(N), N):
            t += 1
            eta = 1 / (t * L)
            if Y[i,0]*sum([a[j]*M[j,i] for j in range(N)]) < 1:
                # a[i] = (1-eta*L)*a[i] + eta*Y[i,0]
                a = (1 - eta * L) * a
                a[i] += eta * Y[i,0]
            else:
                # a[i] = (1 - eta * L) * a[i]
                a = (1 - eta * L) * a
        # print a
        epoch += 1

    return a

# Define the predict_linearSVM(x) function, which uses global trained parameters, w

# def predict_linearSVM(x):
#     return array([-1 if w0 + dot(w, el) <= 0 else 1 for el in x])

def RBF_kernel(gamma):
    return lambda x, y: exp(-1*gamma*linalg.norm(x-y)**2)

def linear_kernel(x,y):
    return dot(x,y)

# kernel_func = linear_kernel
kernel_func = RBF_kernel(100)

M = array([[kernel_func(X[i,:], X[j,:]) for j in range(N)] for i in range(N)])

alpha = run_kernelized_Pegasos(X, Y, L, max_epochs, M)

alpha_indices_nonzero = []

for i in range(N):
    if alpha[i] != 0:
        alpha_indices_nonzero.append(i)

def predict_linearSVM(x):
    # return array([-1 if sum([alpha[i]*kernel_func(X[i,:], el) for i in range(N)]) <= 0 else 1 for el in x])
    return array([-1 if sum([alpha[i]*kernel_func(X[i,:], el) for i in alpha_indices_nonzero]) <= 0 else 1 for el in x])

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

