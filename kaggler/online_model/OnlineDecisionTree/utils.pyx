import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, abs

ctypedef np.int_t DTYPE_t

cpdef DTYPE_t argmax(dict d):
    cdef double max_count = 0
    cdef double total_count = 0
    cdef double value
    cdef DTYPE_t key
    cdef DTYPE_t max_class = 0
    for key, value in d.iteritems():
        total_count += value
        if value > max_count:
            max_count = value
            max_class = key
    return max_class


def predict_max(list a):
    return argmax(count_dict(a))

cpdef dict count_dict(list a):
    cdef DTYPE_t x
    cdef dict d = {}
    for x in a:
        d.setdefault(x, 0)
        d[x] += 1
    return d

cpdef double mean_squared_error(list x):
    cdef np.ndarray xnp
    xnp = np.array(x)
    xnp = xnp - xnp.mean()
    return sqrt((xnp * xnp.T).mean())

cpdef double mean_absolute_error(list x):
    cdef np.ndarray xnp
    xnp = np.array(x)
    xnp = xnp - xnp.mean()
    return abs(xnp).mean()

cpdef double gini(list x):
    cdef dict d = {}
    cdef double total
    cdef list to_square
    cdef np.ndarray to_square2
    cdef DTYPE_t y
    for y in x:
        d.setdefault(y, 0)
        d[y] += 1
    total = len(x)
    to_square = []
    cdef double value
    cdef DTYPE_t key
    for key, value in d.iteritems():
        to_square.append(value/total)
    to_square2 = np.array(to_square)
    return 1 - (to_square2 * to_square2.T).sum()

cpdef tuple bin_split(list sample_feature, double feature_value):
    cdef list left, right
    cdef tuple x
    left = [x[1] for x in sample_feature if x[0]<=feature_value]
    right = [x[1] for x in sample_feature if x[0]>feature_value]
    return left, right
