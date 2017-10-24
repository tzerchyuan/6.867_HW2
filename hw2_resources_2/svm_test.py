import numpy as np
from plotBoundary import *
import pylab as pl
# import your SVM training code
from Problem21 import *

# parameters
name = '4'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()
# X, Y = np.array([[2,2],[2,3],[0,-1],[-3,-2]]), np.array([[1],[1],[-1],[-1]])


# print X
# print(Y.shape)

# Carry out training, primal and/or dual
C = 1
alpha_threshold = 10**(-4)
kernel_func = RBF_kernel(.1)
# kernel_func = linear_kernel
w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, X, Y, None, kernel_func)
# w0, w, sv_indices, etas, alphas, savedX, savedY = run_SVM(C, alpha_threshold, X, Y)
print("W0: ", w0)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

# Define the predictSVM(x) function, which uses trained parameters
# def predictSVM(x):
#     return -1 if np.dot(w, x) + w0 <= 0 else 1

# def predictSVM(x):
#     return -1 if sum([Y[n, 0] * kernel_func(x, X[n,:]) for n in range(Y.shape[0])]) + w0 <= 0 else 1

# Vectorized
def predictSVM(x):
    # return np.array([-1 if sum([alphas[n,0] * savedY[n, 0] * kernel_func(el, savedX[n,:]) for n in range(savedY.shape[0])]) + w0 <= 0 else 1 for el in x])
    return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])
# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
