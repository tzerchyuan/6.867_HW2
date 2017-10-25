import numpy as np
from cvxopt import matrix, solvers
import scipy.linalg

def T(M):
    return np.transpose(M)

def I(n):
    return np.identity(n)

def diag(v):
    return np.diag(v)

# Set C, alpha threshold

C = .5
alpha_threshold = 10**(-4)
X, Y = np.array([[2,2],[2,3],[0,-1],[-3,-2]]), np.array([[1],[1],[-1],[-1]])
# print(X)
# print(Y.shape)
# print(Y[1,0])
# print(Y)
def run_SVM(C, alpha_threshold, XT, Y):
    X = T(XT)
    N = Y.shape[0]

    # matrices for cvxopt

    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i,j] = Y[i,0]*Y[j,0]*np.dot(X[:,i], X[:,j])
    # print("P:", P)
    q = -1*np.ones(N)

    A = T(Y)
    b = 0

    G = np.concatenate((-I(N), I(N)))
    h = np.concatenate((np.zeros(N), C*np.ones(N)))

    P = matrix(P, tc = "d")
    q = matrix(q, tc = "d")
    G = matrix(G, tc = "d")
    h = matrix(h, tc = "d")
    A = A.astype('d')
    # print(A)
    A = matrix(A, tc = "d")
    b = matrix(b, tc = "d")

    # print(P)
    # print(q)
    # print(G)
    # print(h)
    # print(A)
    # print(b)

    # find the solution using cvxopt

    solution = solvers.qp(P, q, G, h, A, b)
    xvals = np.array(solution['x'])

    # print(xvals)
    # print("\nThresholded:\n")
    xvals_thresholded = np.zeros(xvals.shape)

    for i in range(xvals.shape[0]):
        for j in range(xvals.shape[1]):
            xvals_thresholded[i,j] = 0 if abs(xvals[i,j]) < alpha_threshold else xvals[i,j]

    # print(xvals_thresholded)


    # Calculate b, aka w0

    sv_ct = 0
    sv_indices = []
    sv_indices_alpha_under_C = []

    for i in range(xvals_thresholded.shape[0]):
        for j in range(xvals_thresholded.shape[1]):
            if 0 < abs(xvals_thresholded[i,j]):
                sv_ct += 1
                sv_indices.append(i)
                if abs(xvals_thresholded[i,j]) < C:
                    sv_indices_alpha_under_C.append(i)

    print("\nNumber of Support Vectors: " + str(sv_ct))
    # print(sv_indices)
    #
    # w0_sum = 0
    # for n in sv_indices_alpha_under_C:
    #     w0_sum += Y[n]
    #     w0_sum -= sum([xvals_thresholded[m]*Y[m]*np.dot(X[:,n], X[:,m]) for m in sv_indices])
    #
    # w0 = 1/sv_ct * w0_sum
    # print(w0)
    # print("w0 = " + str(w0) + "\n")
    #
    w = sum([xvals_thresholded[n]*Y[n,0]*X[:,n] for n in range(N)])
    # print("W: ", w)

    # Use other LP method to calculate w0

    q2 = np.concatenate(([0], np.ones(N)*C))
    # print "q2:"
    # print q2

    G2a = scipy.linalg.block_diag(np.array([0]), -1*I(N))
    # Y_extended_dim = np.ones((Y.shape[0], 1))
    # for i in range(Y.shape[0]):
    #     Y_extended_dim[i,0] = Y[i]
    G2b = -1*np.concatenate((Y, I(N)), axis = 1)
    # print(Y_extended_dim)
    G2 = np.concatenate((G2a, G2b))
    # print "G2:"
    # print(G2)

    h2a = np.zeros(N+1)
    h2b = [Y[i,0]*np.dot(w, X[:,i])-1 for i in range(N)]
    h2 = np.concatenate((h2a, h2b))
    # print "h2:"
    # print h2

    P2 = matrix(np.zeros((N+1, N+1)))
    q2 = matrix(q2, tc = "d")
    G2 = matrix(G2, tc = "d")
    h2 = matrix(h2, tc = "d")

    # print q2
    # print G2
    # print h2

    # Quadratic Program and Linear Program are equivalent here since this is LP.

    # solution2 = solvers.qp(P2, q2, G2, h2)
    # xvals2 = np.array(solution2['x'])

    solution3 = solvers.lp(q2, G2, h2)
    sol = solution3['x']

    # print "xvals2:"
    # print(xvals2)

    # print "sol:"
    # print sol
    #
    # print("---------------------------------\n")
    # print "w0 = " + str(sol[0])
    # print "w = " + str(w)
    #
    # print("\n---------------------------------\n")
    # print "Support Vectors: "
    # print(str([list(X[:,i]) for i in sv_indices]))

    w0 = sol[0]
    return w0, w, sv_indices, sol[1:], xvals_thresholded, XT, Y

# w0, w, sv_indices, etas, alphas = run_SVM(C, alpha_threshold, X, Y)

# print("\n----\n")
# print w0
# print w
# print [X[:,sv_indices[i]] for i in range(len(sv_indices))]
# print etas
# print alphas


def run_SVM_kernel(C, alpha_threshold, XT, Y, kernel_mat, kernel_func):
    X = T(XT)
    N = Y.shape[0]

    if kernel_mat is not None:
        mat = kernel_mat
    else:
        mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                mat[i,j] = kernel_func(X[:,i], X[:,j])
    # matrices for cvxopt

    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i,j] = Y[i,0]*Y[j,0]*mat[i,j]
    q = -1*np.ones(N)

    A = T(Y)
    b = 0

    G = np.concatenate((-I(N), I(N)))
    h = np.concatenate((np.zeros(N), C*np.ones(N)))

    P = matrix(P, tc = "d")
    q = matrix(q, tc = "d")
    G = matrix(G, tc = "d")
    h = matrix(h, tc = "d")
    A = A.astype('d')
    A = matrix(A, tc = "d")
    b = matrix(b, tc = "d")

    # print(P)
    # print(q)
    # print(G)
    # print(h)
    # print(A)
    # print(b)

    # find the solution using cvxopt

    solution = solvers.qp(P, q, G, h, A, b)
    xvals = np.array(solution['x'])

    # print(xvals)
    # print("\nThresholded:\n")
    xvals_thresholded = np.zeros(xvals.shape)

    for i in range(xvals.shape[0]):
        for j in range(xvals.shape[1]):
            xvals_thresholded[i,j] = 0 if abs(xvals[i,j]) < alpha_threshold else xvals[i,j]

    # print(xvals_thresholded)


    # Calculate b, aka w0

    sv_ct = 0
    sv_indices = []
    sv_indices_alpha_under_C = []

    for i in range(xvals_thresholded.shape[0]):
        for j in range(xvals_thresholded.shape[1]):
            if 0 < abs(xvals_thresholded[i,j]):
                sv_ct += 1
                sv_indices.append(i)
                if abs(xvals_thresholded[i,j]) < C:
                    sv_indices_alpha_under_C.append(i)

    print("\nNumber of Support Vectors: " + str(sv_ct))
    # print(sv_indices)
    #
    # w0_sum = 0
    # for n in sv_indices_alpha_under_C:
    #     w0_sum += Y[n]
    #     w0_sum -= sum([xvals_thresholded[m]*Y[m]*np.dot(X[:,n], X[:,m]) for m in sv_indices])
    #
    # w0 = 1/sv_ct * w0_sum
    # print(w0)
    # print("w0 = " + str(w0) + "\n"


    # Use other LP method to calculate w0

    # We don't use w anymore; we just incorporate it into the kernel
    # w = sum([xvals_thresholded[n]*Y[n,0]*X[:,n] for n in range(N)])
    # print("W: ", w)

    q2 = np.concatenate(([0], np.ones(N)*C))
    # print "q2:"
    # print q2

    G2a = scipy.linalg.block_diag(np.array([0]), -1*I(N))
    # Y_extended_dim = np.ones((Y.shape[0], 1))
    # for i in range(Y.shape[0]):
    #     Y_extended_dim[i,0] = Y[i]
    G2b = -1*np.concatenate((Y, I(N)), axis = 1)
    # print(Y_extended_dim)
    G2 = np.concatenate((G2a, G2b))
    # print "G2:"
    # print(G2)

    h2a = np.zeros(N+1)

    # Product of w and phi(x_n)
    wx_product = [sum([xvals_thresholded[n,0]*Y[n,0]*mat[n,i] for n in range(N)]) for i in range(N)]

    h2b = [Y[i,0]*wx_product[i]-1 for i in range(N)]
    # Original code
    # h2b = [Y[i,0]*np.dot(w, X[:,i])-1 for i in range(N)]
    h2 = np.concatenate((h2a, h2b))
    # print "h2:"
    # print h2

    P2 = matrix(np.zeros((N+1, N+1)))
    q2 = matrix(q2, tc = "d")
    G2 = matrix(G2, tc = "d")
    h2 = matrix(h2, tc = "d")

    # print q2
    # print G2
    # print h2

    # Quadratic Program and Linear Program are equivalent here since this is LP.

    # solution2 = solvers.qp(P2, q2, G2, h2)
    # xvals2 = np.array(solution2['x'])

    solution3 = solvers.lp(q2, G2, h2)
    sol = solution3['x']

    # print "xvals2:"
    # print(xvals2)

    # print "sol:"
    # print sol
    #
    # print("---------------------------------\n")
    # print "w0 = " + str(sol[0])
    # print "w = " + str(w)
    #
    # print("\n---------------------------------\n")
    # print "Support Vectors: "
    # print(str([list(X[:,i]) for i in sv_indices]))

    w0 = sol[0]
    return w0, wx_product, sv_indices, sol[1:], xvals_thresholded, XT, Y


def linear_kernel(x,y):
    return np.dot(x,y)

def RBF_kernel(gamma):
    return lambda x, y: np.exp(-1*np.linalg.norm((x-y))**2*gamma)

# w0, wx_product, sv_indices, etas, alphas = run_SVM_kernel(C, alpha_threshold, X, Y, None, linear_kernel)
