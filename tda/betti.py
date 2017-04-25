import numpy as np
from .matreduce import reduce_matrix


# return the n-simplices in a complex
def nSimplices(n, complex):
    nchain = []
    for simplex in complex:
        if len(simplex) == (n + 1):
            nchain.append(simplex)
    if (nchain == []):
        nchain = [0]
    return nchain

# check if simplex is a face of another simplex


def checkFace(face, simplex):
    if simplex == 0:
        return 1
    elif set(face) < set(simplex):  # if face is a subset of simplex
        return 1
    else:
        return 0

# build boundary matrix for dimension n ---> (n-1) = p


def boundaryMatrix(nchain, pchain):
    bmatrix = np.zeros((len(nchain), len(pchain)))
    i = 0
    for nSimplex in nchain:
        j = 0
        for pSimplex in pchain:
            bmatrix[i, j] = checkFace(pSimplex, nSimplex)
            j += 1
        i += 1
    return bmatrix.T


def betti(complex):
    # get the maximum dimension of the simplicial complex, 2 in our example
    max_dim = len(max(complex, key=len))
    # setup array to store n-th dimensional Betti numbers
    betti_array = np.zeros(max_dim)
    z_n = np.zeros(max_dim)  # number of cycles (from cycle group)
    b_n = np.zeros(max_dim)  # b_(n-1) boundary group
    # loop through each dimension starting from maximum to generate boundary
    # maps
    for i in range(max_dim):
        bm = 0  # setup n-th boundary matrix
        chain2 = nSimplices(i, complex)  # n-th chain group
        if i == 0:  # there is no n+1 boundary matrix in this case
            bm = 0
            z_n[i] = len(chain2)
            b_n[i] = 0
        else:
            chain1 = nSimplices(i - 1, complex)  # (n-1)th chain group
            bm = reduce_matrix(boundaryMatrix(chain2, chain1))
            z_n[i] = bm[2]
            b_n[i] = bm[1]  # b_(n-1)

    for i in range(max_dim):  # Calculate betti number: Z_n - B_n
        if (i + 1) < max_dim:
            betti_array[i] = z_n[i] - b_n[i + 1]
        else:
            # if there are no higher simplices, the boundary group of this
            # chain is 0
            betti_array[i] = z_n[i] - 0

    return betti_array
