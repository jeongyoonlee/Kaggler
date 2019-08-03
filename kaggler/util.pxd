cimport numpy as np


ctypedef np.int_t DTYPE_t

cdef inline double fmax(double a, double b): return a if a >= b else b
cdef inline double fmin(double a, double b): return a if a <= b else b

cdef double sigm(double x)
cpdef double gini(list x)
cpdef DTYPE_t argmax(dict d)
cpdef dict count_dict(list a)
