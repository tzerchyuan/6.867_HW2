from numpy import *
# from plotBoundary import *
# import pylab as pl
import matplotlib.pyplot as plt
from Problem21 import run_SVM_kernel, linear_kernel, RBF_kernel
from lr_test import *
# import data
train_1 = loadtxt('data/mnist_digit_1.csv')
train_7 = loadtxt('data/mnist_digit_7.csv')
X_1 = train_1[:500]
X_7 = train_7[:500]

train_X = concatenate((X_1[:200], X_7[:200]))
print(train_X[:10,:])
train_Y = [1]*200 + [-1]*200
print(len(train_Y))
validate_X = concatenate((X_1[200:350],X_7[200:350]))
validate_Y = [1]*150 + [-1]*150

test_X = concatenate((X_1[350:500], X_7[350:500]))
test_Y = [1]*150 + [-1]*150

# create corresponding sets with normalize pixels
def normalize(X):
    # returns normalized datasets
    new_X = copy.deepcopy(X)
    return new_X*2/255-1

normalized_train_X = normalize(train_X)
normalized_validate_X = normalize(validate_X)
normalized_test_X = normalize(test_X)

C = 1
alpha_threshold = 10**(-8)
# kernel_func = linear_kernel
kernel_func = RBF_kernel(100)
train_Y_np = array(train_Y).reshape((len(train_Y), 1))
w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, normalized_train_X, train_Y_np, None, kernel_func)

print("W0: ", w0)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

def predictSVM(x):
      return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])

# Now we have the decision boundary, let's see how we perform on the test set

test_labels_svm = predictSVM(normalized_test_X)

print(test_labels_svm)
misclassified = [i for i in range(len(test_Y)) if (i < 150 and test_labels_svm[i] == -1) or (i >= 150 and test_labels_svm[i] == 1)]
print("MISCLASSIFIED: ", misclassified)

for img_idx in misclassified:
    plt.imshow(normalized_test_X[img_idx].reshape((28,28)))
    plt.show()
