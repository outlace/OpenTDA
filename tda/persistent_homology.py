import itertools
import functools

import numpy as np
import networkx as nx
from scipy.spatial.distance import squareform, pdist

from .snf import low, smith_normal_form

__all__ = ['PersistentHomology']


def buildGraph(data, epsilon=1., metric='euclidean', p=2):
    D = squareform(pdist(data, metric=metric, p=p))
    D[D >= epsilon] = 0.
    G = nx.Graph(D)
    edges = list(map(set, G.edges()))
    weights = [G.get_edge_data(u, v)['weight'] for u, v in G.edges()]
    return G.nodes(), edges, weights


def lower_nbrs(nodeSet, edgeSet, node):
    return {x for x in nodeSet if {x, node} in edgeSet and node > x}


def rips(nodes, edges, k):
    VRcomplex = [{n} for n in nodes]
    for e in edges:  # add 1-simplices (edges)
        VRcomplex.append(e)
    for i in range(k):
        # skip 0-simplices
        for simplex in [x for x in VRcomplex if len(x) == i + 2]:
            # for each u in simplex
            nbrs = set.intersection(
                *[lower_nbrs(nodes, edges, z) for z in simplex])
            for nbr in nbrs:
                VRcomplex.append(set.union(simplex, {nbr}))
    return VRcomplex


def ripsFiltration(graph, k):
    nodes, edges, weights = graph
    VRcomplex = [{n} for n in nodes]
    filter_values = [0 for j in VRcomplex]  # vertices have filter value of 0
    # add 1-simplices (edges) and associated filter values
    for i in range(len(edges)):
        VRcomplex.append(edges[i])
        filter_values.append(weights[i])
    if k > 1:
        for i in range(k):
            # skip 0-simplices and 1-simplices
            for simplex in [x for x in VRcomplex if len(x) == i + 2]:
                # for each u in simplex
                nbrs = set.intersection(
                    *[lower_nbrs(nodes, edges, z) for z in simplex])
                for nbr in nbrs:
                    newSimplex = set.union(simplex, {nbr})
                    VRcomplex.append(newSimplex)
                    filter_values.append(getFilterValue(
                        newSimplex, VRcomplex, filter_values))

    # sort simplices according to filter values
    return sortComplex(VRcomplex, filter_values)


def getFilterValue(simplex, edges, weights):
    oneSimplices = list(itertools.combinations(simplex, 2))
    max_weight = 0
    for oneSimplex in oneSimplices:
        filter_value = weights[edges.index(set(oneSimplex))]
        if filter_value > max_weight:
            max_weight = filter_value
    return max_weight


def compare(item1, item2):
    if len(item1[0]) == len(item2[0]):
        if item1[1] == item2[1]:
            if sum(item1[0]) > sum(item2[0]):
                return 1
            else:
                return -1
        else:
            if item1[1] > item2[1]:
                return 1
            else:
                return -1
    else:
        if len(item1[0]) > len(item2[0]):
            return 1
        else:
            return -1


def sortComplex(filterComplex, filterValues):
    pairedList = zip(filterComplex, filterValues)
    sortedComplex = sorted(pairedList, key=functools.cmp_to_key(compare))
    sortedComplex = [list(t) for t in zip(*sortedComplex)]

    return sortedComplex


def nSimplices(n, filterComplex):
    nchain = []
    nfilters = []
    for i in range(len(filterComplex[0])):
        simplex = filterComplex[0][i]
        if len(simplex) == (n + 1):
            nchain.append(simplex)
            nfilters.append(filterComplex[1][i])
    if (nchain == []):
        nchain = [0]
    return nchain, nfilters


def checkFace(face, simplex):
    if simplex == 0:
        return 1

    elif (set(face) < set(simplex) and (len(face) == (len(simplex) - 1))):
        return 1
    else:
        return 0


def filterBoundaryMatrix(filterComplex):
    bmatrix = np.zeros(
        (len(filterComplex[0]), len(filterComplex[0])), dtype=np.uint8)

    i = 0
    for colSimplex in filterComplex[0]:
        j = 0
        for rowSimplex in filterComplex[0]:
            bmatrix[j, i] = checkFace(rowSimplex, colSimplex)
            j += 1
        i += 1
    return bmatrix


def readIntervals(reduced_matrix, filterValues):
    intervals = []
    m = reduced_matrix.shape[1]
    for j in range(m):
        low_j = low(j, reduced_matrix)
        if low_j == (m - 1):
            interval_start = [j, -1]
            intervals.append(interval_start)

        else:
            feature = intervals.index([low_j, -1])
            intervals[feature][1] = j
            epsilon_start = filterValues[intervals[feature][0]]
            epsilon_end = filterValues[j]
            if epsilon_start == epsilon_end:
                intervals.remove(intervals[feature])

    return intervals


def readPersistence(intervals, filterComplex):
    persistence = []
    for interval in intervals:
        start = interval[0]
        end = interval[1]

        homology_group = (len(filterComplex[0][start]) - 1)
        epsilon_start = filterComplex[1][start]
        epsilon_end = filterComplex[1][end]
        persistence.append([homology_group, [epsilon_start, epsilon_end]])

    return persistence


class PersistentHomology(object):

    def __init__(self, epsilon=1., k=3):
        self.epsilon = epsilon
        self.k = k

    def fit(self, X):
        self.graph = buildGraph(X, epsilon=self.epsilon)
        self.ripsComplex = ripsFiltration(self.graph, k=self.k)
        self.boundary_matrix = filterBoundaryMatrix(self.ripsComplex)
        self.reduced_boundary_matrix = smith_normal_form(self.boundary_matrix)
        return self

    def transform(self, X):
        intervals = readIntervals(self.reduced_boundary_matrix,
                                  self.ripsComplex[1])
        persistence = readPersistence(intervals, self.ripsComplex)
        return persistence

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
