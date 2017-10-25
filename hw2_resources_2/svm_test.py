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

for C in [0.01, 0.1, 1, 10, 100]:
    for g in [0.01, 0.1, 1, 10]:
        # Carry out training, primal and/or dual
        # C = 0.1
        alpha_threshold = 10**(-4)
        kernel_func = RBF_kernel(g)
        # kernel_func = linear_kernel
        w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, X, Y, None, kernel_func)
        print("W0: ", w0)

        alphas_indices_nonzero = []

        for i in range(savedY.shape[0]):
            if alphas[i] != 0:
                alphas_indices_nonzero.append(i)

        # margin size
        print('margin---> ', 1.0/(math.sqrt(np.sum(alphas_indices_nonzero))))
        print('C', c)
        print('g', g)
        # Vectorize
        def predictSVM(x):
            return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])
        # plot training results
        plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train with C =' + str(C) + ' Gamma = '+ str(g))


        # print '======Validation======'
        # # load data from csv files
        # validate = loadtxt('data/data'+name+'_validate.csv')
        # X = validate[:, 0:2]
        # Y = validate[:, 2:3]
        # # plot validation results
        # plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate with C =' + str(C)+ ' Gamma = '+ str(g))

pl.show()
