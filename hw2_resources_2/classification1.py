from numpy import *
# import data
train_1 = loadtxt('data/mnist_digit_1.csv')
train_7 = loadtxt('data/mnist_digit_7.csv')
X_1 = train_1[:500]
X_7 = train_7[:500]
train_X = X_1[:200] + X_7[:200]
train_Y = [1 for i in range(200)] + [-1 for i in range(200)]

validate_X = X_1[200:350] + X_7[200:350]
validate_Y = [1 for i in range(150)] + [-1 for i in range(150)]

test_X = X_1[350:500] + X_7[350:500]
test_y = [1 for i in range(150)] + [-1 for i in range(150)]
