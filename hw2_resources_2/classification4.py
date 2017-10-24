from numpy import *
# from plotBoundary import *
# import pylab as pl
import matplotlib.pyplot as plt
from Problem21 import run_SVM_kernel, linear_kernel, RBF_kernel
from lr_test import *

# import data
train_0 = loadtxt('data/mnist_digit_0.csv')
train_1 = loadtxt('data/mnist_digit_1.csv')
train_2 = loadtxt('data/mnist_digit_2.csv')
train_3 = loadtxt('data/mnist_digit_3.csv')
train_4 = loadtxt('data/mnist_digit_4.csv')
train_5 = loadtxt('data/mnist_digit_5.csv')
train_6 = loadtxt('data/mnist_digit_6.csv')
train_7 = loadtxt('data/mnist_digit_7.csv')
train_8 = loadtxt('data/mnist_digit_8.csv')
train_9 = loadtxt('data/mnist_digit_9.csv')
X_0 = train_0[:500]
X_1 = train_1[:500]
X_2 = train_2[:500]
X_3 = train_3[:500]
X_4 = train_4[:500]
X_5 = train_5[:500]
X_6 = train_6[:500]
X_7 = train_7[:500]
X_8 = train_8[:500]
X_9 = train_9[:500]

train_X = concatenate((X_0[:200], X_2[:200],X_4[:200],X_6[:200],X_8[:200],X_1[:200],X_3[:200],X_5[:200],X_7[:200],X_9[:200]))
train_Y = [1] * 1000 + [-1] * 1000
validate_X = concatenate((X_0[200:350], X_2[200:350],X_4[200:350],X_6[200:350],X_8[200:350],X_1[200:350],X_3[200:350],X_5[200:350],X_7[200:350],X_9[200:350]))
validate_Y = [1] * 750 + [-1] * 750

test_X = concatenate((X_0[350:500], X_2[350:500],X_4[350:500],X_6[350:500],X_8[350:500],X_1[350:500],X_3[350:500],X_5[350:500],X_7[350:500],X_9[350:500]))
test_Y = [1] * 750 + [-1] * 750


# create corresponding sets with normalize pixels
def normalize(X):
    # returns normalized datasets
    new_X = copy.deepcopy(X)
    return new_X * 2 / 255 - 1


normalized_train_X = normalize(train_X)
normalized_validate_X = normalize(validate_X)
normalized_test_X = normalize(test_X)

######################### LOGISTIC REGRESSION ############################
# # add bias term for samples
# def predict_lr(x, w):
#     result = []
#     for sample in x:
#         sample = np.insert(sample, 0, 1)
#         prob = dot_and_sigmoid(sample, w)
#         if prob > 0.5:
#             result.append(1)
#         else:
#             result.append(-1)
#     return np.array(result)
#
# normalized_train_X_lr = transform_x(normalized_train_X)
# w = new_sgd(normalized_train_X_lr, train_Y, 0.001, 0.02, 1, 2)
# test_labels_lr = predict_lr(normalized_test_X, w)
#
#
# # off the shelf implementation
# # l = LogisticRegression(penalty='l2', C=0.01)
# # l.fit(normalized_train_X, train_Y)
# #
# # test_labels_lr = l.predict(normalized_test_X)
#
#
# misclassified = [i for i in range(len(test_Y)) if (i < 150 and test_labels_lr[i] == -1) or (i >= 150 and test_labels_lr[i] == 1)]
# print("MISCLASSIFIED: ", misclassified)
# # print(len(misclassified))
#
# if len(misclassified) <10:
#     for img_idx in misclassified:
#         plt.imshow(normalized_test_X[img_idx].reshape((28,28)))
#         plt.show()

######################## Linear SVM ############################################

# C = 1
alpha_threshold = 10**(-8)
C_range = [3**(i) for i in range(-5,6)]

kernel_func = linear_kernel
# kernel_func = RBF_kernel(100)
train_Y_np = array(train_Y).reshape((len(train_Y), 1))

best_param = C_range[0]
best_validation_error = 1

for C in C_range:
    w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, normalized_train_X, train_Y_np, None, kernel_func)

    alphas_indices_nonzero = []

    for i in range(savedY.shape[0]):
        if alphas[i] != 0:
            alphas_indices_nonzero.append(i)

    def predictSVM(x):
          return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])


    # Validation step

    validation_labels_svm = predictSVM(normalized_validate_X)

    misclassified = [i for i in range(len(validate_Y)) if (i < 750 and validation_labels_svm[i] == -1) or (i >= 750 and validation_labels_svm[i] == 1)]

    print("-------------------------------------------------------")
    print((C, alpha_threshold))
    print("MISCLASSIFIED ON VALIDATION: ", misclassified)

    validation_error = 1.0*len(misclassified)/len(validation_labels_svm)

    if validation_error < best_validation_error:
        best_validation_error = validation_error
        best_param = C

print("-------------------------------------------------------")
print "Best Params after Validation: ", best_param
print "Best Validation error: ", best_validation_error

C = best_param

# Test Phase

w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, normalized_train_X,
                                                                          train_Y_np, None, kernel_func)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

def predictSVM(x):
    return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])

test_labels_svm = predictSVM(normalized_test_X)

# print(test_labels_svm)
misclassified = [i for i in range(len(test_Y)) if (i < 750 and test_labels_svm[i] == -1) or (i >= 750 and test_labels_svm[i] == 1)]
print("MISCLASSIFIED TEST: ", misclassified)

for img_idx in misclassified:
    plt.imshow(normalized_test_X[img_idx].reshape((28,28)))
    plt.show()


######################### Linear SVM Non-Normalized ############################################

# C = 1
alpha_threshold = 10**(-8)
C_range = [3**(i) for i in range(-5,6)]

kernel_func = linear_kernel
# kernel_func = RBF_kernel(100)
train_Y_np = array(train_Y).reshape((len(train_Y), 1))

best_param = C_range[0]
best_validation_error = 1

for C in C_range:
    w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, train_X, train_Y_np, None, kernel_func)

    alphas_indices_nonzero = []

    for i in range(savedY.shape[0]):
        if alphas[i] != 0:
            alphas_indices_nonzero.append(i)

    def predictSVM(x):
          return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])

    # Validation step

    validation_labels_svm = predictSVM(validate_X)

    misclassified = [i for i in range(len(validate_Y)) if (i < 750 and validation_labels_svm[i] == -1) or (i >= 750 and validation_labels_svm[i] == 1)]

    print("-------------------------------------------------------")
    print(C)
    print("MISCLASSIFIED ON VALIDATION: ", misclassified)

    validation_error = 1.0*len(misclassified)/len(validation_labels_svm)

    if validation_error < best_validation_error:
        best_validation_error = validation_error
        best_param = C

print("-------------------------------------------------------")
print "Best Params after Validation: ", best_param
print "Best Validation error: ", best_validation_error

C = best_param


# Test Phase

w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, train_X,
                                                                          train_Y_np, None, kernel_func)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

def predictSVM(x):
    return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])

test_labels_svm = predictSVM(test_X)

# print(test_labels_svm)
misclassified = [i for i in range(len(test_Y)) if (i < 750 and test_labels_svm[i] == -1) or (i >= 750 and test_labels_svm[i] == 1)]
print("MISCLASSIFIED TEST: ", misclassified)

for img_idx in misclassified:
    plt.imshow(test_X[img_idx].reshape((28,28)))
    plt.show()

####################### Gaussian SVM Normalized ############################################

C_range = [3**(i) for i in range(-5,6)]
alpha_threshold = 10**(-8)
gamma_range = [.01, .1, 1, 10, 100]

# kernel_func = linear_kernel
# kernel_func = RBF_kernel(100)
train_Y_np = array(train_Y).reshape((len(train_Y), 1))

best_params = (C_range[0], gamma_range[0])
best_validation_error = 1

for C in C_range:
    for gamma in gamma_range:
        kernel_func = RBF_kernel(gamma)

        w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, normalized_train_X, train_Y_np, None, kernel_func)

        alphas_indices_nonzero = []

        for i in range(savedY.shape[0]):
            if alphas[i] != 0:
                alphas_indices_nonzero.append(i)

        def predictSVM(x):
              return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])


        # Validation step

        validation_labels_svm = predictSVM(normalized_validate_X)

        misclassified = [i for i in range(len(validate_Y)) if (i < 750 and validation_labels_svm[i] == -1) or (i >= 750 and validation_labels_svm[i] == 1)]

        print("-------------------------------------------------------")
        print((C, gamma))
        print("MISCLASSIFIED ON VALIDATION: ", misclassified)

        validation_error = 1.0*len(misclassified)/len(validation_labels_svm)

        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_params = (C, gamma)

print("-------------------------------------------------------")
print "Best Params after Validation: ", best_params
print "Best Validation error: ", best_validation_error

C, gamma = best_params


# Test Phase

kernel_func = RBF_kernel(gamma)
w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, normalized_train_X,
                                                                          train_Y_np, None, kernel_func)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

def predictSVM(x):
    return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])

test_labels_svm = predictSVM(normalized_test_X)

# print(test_labels_svm)
misclassified = [i for i in range(len(test_Y)) if (i < 750 and test_labels_svm[i] == -1) or (i >= 750 and test_labels_svm[i] == 1)]
print("MISCLASSIFIED TEST: ", misclassified)

for img_idx in misclassified:
    plt.imshow(normalized_test_X[img_idx].reshape((28,28)))
    plt.show()


# ######################### Gaussian SVM Non-Normalized ############################################

C_range = [3**(i) for i in range(-5,6)]
alpha_threshold = 0
gamma_range = [.01, .1, 1, 10, 100]

# kernel_func = linear_kernel
# kernel_func = RBF_kernel(100)
train_Y_np = array(train_Y).reshape((len(train_Y), 1))

best_params = (C_range[0], gamma_range[0])
best_validation_error = 1

for C in C_range:
    for gamma in gamma_range:
        kernel_func = RBF_kernel(gamma)

        w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, train_X, train_Y_np, None, kernel_func)

        alphas_indices_nonzero = []

        for i in range(savedY.shape[0]):
            if alphas[i] != 0:
                alphas_indices_nonzero.append(i)

        def predictSVM(x):
              return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])


        # Validation step

        validation_labels_svm = predictSVM(validate_X)

        misclassified = [i for i in range(len(validate_Y)) if (i < 750 and validation_labels_svm[i] == -1) or (i >= 750 and validation_labels_svm[i] == 1)]

        print("-------------------------------------------------------")
        print((C, gamma))
        print("MISCLASSIFIED ON VALIDATION: ", misclassified)

        validation_error = 1.0*len(misclassified)/len(validation_labels_svm)

        if validation_error < best_validation_error:
            best_validation_error = validation_error
            best_params = (C, gamma)

print("-------------------------------------------------------")
print "Best Params after Validation: ", best_params
print "Best Validation error: ", best_validation_error

C, gamma = best_params


# Test Phase

kernel_func = RBF_kernel(gamma)
w0, wx_product, sv_indices, etas, alphas, savedX, savedY = run_SVM_kernel(C, alpha_threshold, train_X,
                                                                          train_Y_np, None, kernel_func)

alphas_indices_nonzero = []

for i in range(savedY.shape[0]):
    if alphas[i] != 0:
        alphas_indices_nonzero.append(i)

def predictSVM(x):
    return np.array([-1 if sum([alphas[n, 0] * savedY[n, 0] * kernel_func(el, savedX[n, :]) for n in alphas_indices_nonzero]) + w0 <= 0 else 1 for el in x])

test_labels_svm = predictSVM(test_X)

# print(test_labels_svm)
misclassified = [i for i in range(len(test_Y)) if (i < 750 and test_labels_svm[i] == -1) or (i >= 750 and test_labels_svm[i] == 1)]
print("MISCLASSIFIED TEST: ", misclassified)

for img_idx in misclassified:
    plt.imshow(test_X[img_idx].reshape((28,28)))
    plt.show()