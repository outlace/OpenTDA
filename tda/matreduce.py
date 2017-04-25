import numpy as np
from numpy.linalg import svd


def reduce_matrix(A, eps=None):
    if np.size(A) == 0:
        return A, 0, 0
    if np.size(A) == 1:
        return A, 1, []

    m, n = A.shape
    if m != n:
        M = np.zeros(2 * (max(A.shape), ))
        M[:m, :n] = A
    else:
        M = A

    u, s, v = svd(M)
    if eps is None:
        eps = s.max() * max(M.shape) * np.finfo(s.dtype).eps

    null_mask = (s <= eps)

    rank = sum(~null_mask)
    null_space = v[null_mask]

    u = u[~null_mask][:, ~null_mask]
    s = np.diag(s[~null_mask])
    v = v[~null_mask]
    reduced = u.dot(s.dot(v))

    return reduced, rank, null_space
