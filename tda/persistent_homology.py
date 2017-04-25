import numpy as np
import itertools
import functools
import matplotlib.pyplot as plt


# this is the default metric we use but you can use whatever distance
# function you want
def euclidianDist(a, b):
    return np.linalg.norm(a - b)  # euclidian distance metric

# Build neighorbood graph


def buildGraph(raw_data, epsilon=3.1, metric=euclidianDist):  # raw_data is a numpy array
    # initialize node set, reference indices from original data array
    nodes = [x for x in range(raw_data.shape[0])]
    edges = []  # initialize empty edge array
    # initialize weight array, stores the weight (which in this case is the
    # distance) for each edge
    weights = []
    for i in range(raw_data.shape[0]):  # iterate through each data point
        # inner loop to calculate pairwise point distances
        for j in range(raw_data.shape[0] - i):
            a = raw_data[i]
            # each simplex is a set (no order), hence [0,1] = [1,0]; so only
            # store one
            b = raw_data[j + i]
            if (i != j + i):
                dist = metric(a, b)
                if dist <= epsilon:
                    # add edge if distance between points is < epsilon
                    edges.append({i, j + i})
                    weights.append(dist)
    return nodes, edges, weights


# lowest neighbors based on arbitrary ordering of simplices
def lower_nbrs(nodeSet, edgeSet, node):
    return {x for x in nodeSet if {x, node} in edgeSet and node > x}


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
    orderValues = [x for x in range(len(filterComplex))]
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
        (len(filterComplex[0]), len(filterComplex[0])), dtype='>i8')
    # bmatrix[0,:] = 0 #add "zero-th" dimension as first row/column, makes algorithm easier later on
    #bmatrix[:,0] = 0
    i = 0
    for colSimplex in filterComplex[0]:
        j = 0
        for rowSimplex in filterComplex[0]:
            bmatrix[j, i] = checkFace(rowSimplex, colSimplex)
            j += 1
        i += 1
    return bmatrix

# returns row index of lowest "1" in a column i in the boundary matrix


def low(i, matrix):
    col = matrix[:, i]
    col_len = len(col)
    # loop through column from bottom until you find the first 1
    for i in range((col_len - 1), -1, -1):
        if col[i] == 1:
            return i
    # if no lowest 1 (e.g. column of all zeros), return -1 to be 'undefined'
    return -1

# checks if the boundary matrix is fully reduced


def isReduced(matrix):
    for j in range(matrix.shape[1]):  # iterate through columns
        for i in range(j):  # iterate through columns before column j
            low_j = low(j, matrix)
            low_i = low(i, matrix)
            if (low_j == low_i and low_j != -1):
                return i, j  # return column i to add to column j
    return [0, 0]

# the main function to iteratively reduce the boundary matrix


def reduceBoundaryMatrix(matrix):
    # this refers to column index in the boundary matrix
    reduced_matrix = matrix.copy()
    matrix_shape = reduced_matrix.shape
    # this matrix will store the column additions we make
    memory = np.identity(matrix_shape[1], dtype='>i8')
    r = isReduced(reduced_matrix)
    while (r != [0, 0]):
        i = r[0]
        j = r[1]
        col_j = reduced_matrix[:, j]
        col_i = reduced_matrix[:, i]
        # print("Mod: add col %s to %s \n" % (i+1,j+1)) #Uncomment to see what
        # mods are made
        reduced_matrix[:, j] = np.bitwise_xor(
            col_i, col_j)  # add column i to j
        memory[i, j] = 1
        r = isReduced(reduced_matrix)
    return reduced_matrix, memory


# reduced_matrix includes the reduced boundary matrix AND the memory matrix
def readIntervals(reduced_matrix, filterValues):
    # store intervals as a list of 2-element lists, e.g. [2,4] = start at "time" point 2, end at "time" point 4
    # note the "time" points are actually just the simplex index number for
    # now. we will convert to epsilon value later
    intervals = []
    # loop through each column j
    # if low(j) = -1 (undefined, all zeros) then j signifies the birth of a new feature j
    # if low(j) = i (defined), then j signifies the death of feature i
    # for each column (its a square matrix so doesn't matter...)
    for j in range(reduced_matrix[0].shape[1]):
        low_j = low(j, reduced_matrix[0])
        if low_j == -1:
            interval_start = [j, -1]
            # -1 is a temporary placeholder until we update with death time
            intervals.append(interval_start)
            # if no death time, then -1 signifies feature has no end (start -> infinity)
            #-1 turns out to be very useful because in python if we access the list x[-1] then that will return the
            # last element in that list. in effect if we leave the end point of an interval to be -1
            # then we're saying the feature lasts until the very end
        else:  # death of feature
            # find the feature [start,end] so we can update the end point
            feature = intervals.index([low_j, -1])
            intervals[feature][1] = j  # j is the death point
            # if the interval start point and end point are the same, then this feature begins and dies instantly
            # so it is a useless interval and we dont want to waste memory
            # keeping it
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
