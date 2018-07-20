import numpy as np
import sigstuff as sig
import networkx as nx
import cvxpy as cvx

def rigiditymatrix(pos, edat):
    dim = len(pos[0])
    numverts = len(pos)
    R = np.array([]).reshape(0, dim*numverts)
    for e in edat:
        part1 = e[0]
        part2 = e[1]
        dvec = pos[part2] - pos[part1]
        row = np.zeros(dim*numverts)
        row[dim*(part1):dim*(part1)+dim] = -dvec
        row[dim*(part2):dim*(part2)+dim] = dvec
        R = np.vstack((R, row))
    return R

def incidencematix(edat):
    numverts = max([max(e) for e in edat])+1
    I = np.array([]).reshape(0, numverts)
    for e in edat:
        part1 = e[0]
        part2 = e[1]
        row = np.zeros(numverts)
        row[part1] = -1
        row[part2] = 1
        I = np.vstack((I, row))
    return I

def stressmatrix(stress, edat):
    inc = incidencematix(edat)
    diag = np.diag(stress)
    return inc.T.dot(diag.dot(inc))

def stressbasis(pos, edat):
    R = rigiditymatrix(pos, edat)
    return sig.nullspace(R.T).T

def stressmatbasis(pos, edat):
    sbasis = stressbasis(pos, edat)
    return [stressmatrix(s, edat) for s in sbasis]

def augmented_config(P):
    n,d = P.shape
    return np.block([P,np.ones((n,1))])

def adjacency_matrix(G,bar=False):
    if bar:
        G = nx.complement(G)
    return nx.adj_matrix(G).todense().astype(np.float64)

def zeros_constraints(X,A):
    n,_ = X.shape
    return [ X[i,j]*A[i,j] == 0 for i in range(n) for j in range(n) ]

def find_psd_stress(edge_list,P,is_Phat=False,**kwargs):
    """
    Tries to find a PSD stress for dimension d.
    
    edge_list is a list of edges [i,j]
    P is an n x d configuration matrix or an n x (d+1) augmented configuration matrix
    """
    if not is_Phat:
        P = augmented_config(P)
    G = nx.Graph()
    G.add_edges_from(edge_list)
    Abar = adjacency_matrix(G,True)
    
    n,dplusone = P.shape
    X = cvx.Variable((n,n),PSD=True)
    obj = cvx.Maximize(cvx.lambda_sum_smallest(X, dplusone+1))
    # prob = cvx.Problem(obj, [cvx.trace(X) == n,X*P == np.zeros((n,dplusone))] + zeros_constraints(X,Abar))
    prob = cvx.Problem(obj, [cvx.trace(X) == n,X*P == np.zeros((n,dplusone)),cvx.multiply(X,Abar)== np.zeros((n,n))])
    prob.solve(**kwargs)
    return X.value, prob.value, prob.status
    