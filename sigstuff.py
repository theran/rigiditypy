import numpy as np
import cvxpy as cvx

def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def signature(A, atol=1e-13):
    eigs = np.linalg.eigvals(A)
    eigs = [np.sign(eig) if abs(eig)>atol else 0 for eig in eigs]
    numpos = eigs.count(1)
    numneg = eigs.count(-1)
    return (numpos, len(eigs)-numpos-numneg, numneg)

def findPSDlincomb(mats, nullity, **kwdargs):
    adim = len(mats)
    (dim, dim2) = np.shape(mats[0])
    assert dim == dim2, "Matrices are not square; shape of first matrix in list is: "+str((dim, dim2))
    X = cvx.Variable((dim, dim), PSD=True)
    a = cvx.Variable((adim))
    rhs = sum(a[j]*mats[j] for j in range(adim))
    con = [X == rhs, cvx.trace(X) == 1]
    obj = cvx.Maximize(cvx.lambda_sum_smallest(X, nullity+1))
    prob = cvx.Problem(obj, con)
    prob.solve(**kwdargs)
    return X, a, prob
