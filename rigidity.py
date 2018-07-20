import numpy as np
import rigiditypy.sigstuff as sig
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
    constraints = [cvx.trace(X) == n, X*P == np.zeros((n,dplusone)),cvx.multiply(X,Abar) == np.zeros((n,n))]
    prob = cvx.Problem(obj, constraints)
    prob.solve(**kwargs)
    return X.value, prob.value, prob.status, prob

def has_stretched_cycle(edge_list, P):
    """
    Checks if a graph on the line has a stretched cycle

    edge_list is a list of edges [i,j]
    P is a n x 1 configuration matrix
    """
    # form digraph oriented left to right using P
    G = nx.DiGraph()
    G.add_nodes_from(range(len(P)))
    for e in edge_list:
        e = list(e)
        if P[e[1]] < P[e[0]]:
            e.reverse()
        G.add_edge(*e)
    
    # remove each edge and see if there's another directed path
    for e in G.edges():
        G.remove_edge(*e)
        if nx.has_path(G, *e):
            return True
        G.add_edge(*e)
    return False