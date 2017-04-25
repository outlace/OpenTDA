# Author: Carlos Xavier Hernandez <cxh@stanford.edu>

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cimport cython

DTYPE = np.uint8

ctypedef np.uint8_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def low(int i, np.ndarray[DTYPE_t, ndim=2] matrix):
    cdef np.ndarray[DTYPE_t, ndim=1] col
    cdef int j

    col = matrix[:, i]
    j = col.shape[0] - 1
    while j > -1:
        if col[j] == 1:
            return j
        j -= 1
    return col.shape[0] - 1


@cython.boundscheck(False)
@cython.wraparound(False)
def isReduced(np.ndarray[DTYPE_t, ndim=2] matrix, int bound):
    cdef int i, j, low_j, low_i
    cdef int *low_js
    low_js = <int *>malloc((bound+1)*cython.sizeof(int))

    for j in range(bound + 1):
        low_js[j] = low(j, matrix)

    for j in range(matrix.shape[1]):
        low_j = low_js[j]
        for i in range(j):  # iterate through columns before column j
            low_i = low_js[i]
            if (low_j == low_i and low_j != bound):
                return i, j  # return column i to add to column j
    return 0, 0


@cython.boundscheck(False)
@cython.wraparound(False)
def bitwise_xor(np.ndarray[DTYPE_t, ndim=1] col_i,
                np.ndarray[DTYPE_t, ndim=1] col_j,
                np.ndarray[DTYPE_t, ndim=1] out):
    cdef int i, n

    n = out.shape[0] - 1
    while n > -1:
        if (col_i[n] + col_j[n]) == 1:
            out[n] = 1
        n -= 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def smith_normal_form(np.ndarray[DTYPE_t, ndim=2] matrix):
    cdef int i, j, n, m
    cdef np.ndarray[DTYPE_t, ndim=1] col_i, col_j, out
    cdef np.ndarray[DTYPE_t, ndim=2] reduced_matrix

    reduced_matrix = matrix.copy()
    m = reduced_matrix.shape[0]

    i, j = isReduced(reduced_matrix, m - 1)
    while (i != 0 and j != 0):
        col_j = reduced_matrix[:, j]
        col_i = reduced_matrix[:, i]

        out = np.zeros(m, dtype=DTYPE)
        reduced_matrix[:, j] = bitwise_xor(col_i, col_j, out)
        i, j = isReduced(reduced_matrix, m - 1)
    return reduced_matrix
