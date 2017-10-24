from numpy import *
from
from Problem21 import run_SVM_kernel, linear_kernel, RBF_kernel
# import data
train_1 = loadtxt('data/mnist_digit_1.csv')
train_7 = loadtxt('data/mnist_digit_7.csv')
X_1 = train_1[:500]
X_7 = train_7[:500]
train_X = (X_1[:200] + X_7[:200])*2/255 - 1
train_Y = [1 for i in range(200)] + [-1 for i in range(200)]

validate_X = (X_1[200:350] + X_7[200:350])*2/255 - 1
validate_Y = [1 for i in range(150)] + [-1 for i in range(150)]

test_X = (X_1[350:500] + X_7[350:500])*2/255 - 1
test_Y = [1 for i in range(150)] + [-1 for i in range(150)]


C = 1
alpha_threshold = 10**(-8)
kernel_func = linear_kernel
# kernel_func = RBF_kernel(0.1)
w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, train_X, train_Y, None, kernel_func)

print("W0: ", w0)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

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