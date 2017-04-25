import itertools
import functools

import numpy as np
import networkx as nx
from scipy.spatial.distance import squareform, pdist

import matplotlib.pyplot as plt

from .snf import low


# Build neighorbood graph
def buildGraph(data, epsilon=1., metric='euclidean', p=2):
    D = squareform(pdist(data, metric=metric, p=p))
    D[D >= epsilon] = 0.
    G = nx.Graph(D)
    edges = list(map(set, G.edges()))
    weights = [G.get_edge_data(u, v)['weight'] for u, v in G.edges()]
    return G.nodes(), edges, weights


# lowest neighbors based on arbitrary ordering of simplices
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


# k is the maximal dimension we want to compute (minimum is 1, edges)
def ripsFiltration(graph, k):
    nodes, edges, weights = graph
    VRcomplex = [{n} for n in nodes]
    filter_values = [0 for j in VRcomplex]  # vertices have filter value of 0
    for i in range(len(edges)):  # add 1-simplices (edges) and associated filter values
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


# filter value is the maximum weight of an edge in the simplex
def getFilterValue(simplex, edges, weights):
    # get set of 1-simplices in the simplex
    oneSimplices = list(itertools.combinations(simplex, 2))
    max_weight = 0
    for oneSimplex in oneSimplices:
        filter_value = weights[edges.index(set(oneSimplex))]
        if filter_value > max_weight:
            max_weight = filter_value
    return max_weight


def compare(item1, item2):
    # comparison function that will provide the basis for our total order on the simpices
    # each item represents a simplex, bundled as a list [simplex, filter
    # value] e.g. [{0,1}, 4]
    if len(item1[0]) == len(item2[0]):
        if item1[1] == item2[1]:  # if both items have same filter value
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


# need simplices in filtration have a total order
def sortComplex(filterComplex, filterValues):
    # sort simplices in filtration by filter values
    pairedList = zip(filterComplex, filterValues)
    # since I'm using Python 3.5+, no longer supports custom compare, need
    # conversion helper function..its ok
    sortedComplex = sorted(pairedList, key=functools.cmp_to_key(compare))
    sortedComplex = [list(t) for t in zip(*sortedComplex)]
    # then sort >= 1 simplices in each chain group by the arbitrary total
    # order on the vertices
    # orderValues = [x for x in range(len(filterComplex))]
    return sortedComplex


# return the n-simplices and weights in a complex
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


# check if simplex is a face of another simplex
def checkFace(face, simplex):
    if simplex == 0:
        return 1
    # if face is a (n-1) subset of simplex
    elif (set(face) < set(simplex) and (len(face) == (len(simplex) - 1))):
        return 1
    else:
        return 0


# build boundary matrix for dimension n ---> (n-1) = p
def filterBoundaryMatrix(filterComplex):
    bmatrix = np.zeros(
        (len(filterComplex[0]), len(filterComplex[0])), dtype=np.uint8)
    # bmatrix[0,:] = 0 #add "zero-th" dimension as first row/column, makes algorithm easier later on
    # bmatrix[:,0] = 0
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
    m = reduced_matrix[0].shape[1]
    for j in range(m):
        low_j = low(j, reduced_matrix[0])
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
    # this converts intervals into epsilon format and figures out which
    # homology group each interval belongs to
    persistence = []
    for interval in intervals:
        start = interval[0]
        end = interval[1]
        # filterComplex is a list of lists [complex, filter values]
        homology_group = (len(filterComplex[0][start]) - 1)
        epsilon_start = filterComplex[1][start]
        epsilon_end = filterComplex[1][end]
        persistence.append([homology_group, [epsilon_start, epsilon_end]])

    return persistence


def graph_barcode(persistence, homology_group=0):
    # this function just produces the barcode graph for each homology group
    xstart = [s[1][0] for s in persistence if s[0] == homology_group]
    xstop = [s[1][1] for s in persistence if s[0] == homology_group]
    y = [0.1 * x + 0.1 for x in range(len(xstart))]
    plt.hlines(y, xstart, xstop, color='b', lw=4)
    # Setup the plot
    ax = plt.gca()
    plt.ylim(0, max(y) + 0.1)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.xlabel('epsilon')
    plt.ylabel("Betti dim %s" % (homology_group,))
    plt.show()


def drawComplex(origData, ripsComplex, axes=[-6, 8, -6, 6]):
    plt.clf()
    plt.axis(axes)  # axes = [x1, x2, y1, y2]
    plt.scatter(origData[:, 0], origData[:, 1])  # plotting just for clarity
    for i, txt in enumerate(origData):
        plt.annotate(i, (origData[i][0] + 0.05, origData[i][1]))  # add labels

    # add lines for edges
    for edge in [e for e in ripsComplex if len(e) == 2]:
        # print(edge)
        pt1, pt2 = [origData[pt] for pt in [n for n in edge]]
        # plt.gca().add_line(plt.Line2D(pt1,pt2))
        line = plt.Polygon([pt1, pt2], closed=None, fill=None, edgecolor='r')
        plt.gca().add_line(line)

    # add triangles
    for triangle in [t for t in ripsComplex if len(t) == 3]:
        pt1, pt2, pt3 = [origData[pt] for pt in [n for n in triangle]]
        line = plt.Polygon([pt1, pt2, pt3], closed=False,
                           color="blue", alpha=0.3, fill=True, edgecolor=None)
        plt.gca().add_line(line)
    plt.show()
