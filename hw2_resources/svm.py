from cvxopt import matrix, solvers
import numpy as np

X = np.array([[2.0, 2.0],[2.0, 3.0],[0.0, -1.0],[-3.0, -2.0]])
Y = np.array([1.0, 1.0, -1.0, -1.0])

# construct Q matrix
temp = []
for i in range(len(X)):
    row = []
    for j in range(len(X)):
        row.append(Y[i]*Y[j]*np.dot(X[i],X[j]))
    temp.append(row)

P = matrix(temp)
print(P)
print([-1 for i in range(len(X))])
q = matrix([-1.0 for i in range(len(X))])
a = np.zeros((4, 4), float)
c = np.zeros((4, 4), float)
np.fill_diagonal(a, -1)
np.fill_diagonal(c, 1.0)
G = matrix(np.append(a, c))
print(G)
asd
h = matrix([0.0 for i in range(len(X))])
A = matrix([[i] for i in Y])
b = matrix([0.0])

solution = solvers.qp(P, q, G, h, A, b)
xvals = np.array(solution['x'])
