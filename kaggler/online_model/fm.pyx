from __future__ import division
from itertools import izip
import numpy as np
import random

cimport cython
from libc.math cimport exp, sqrt, fabs
cimport numpy as np


np.import_array()

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b


cdef double sigm(double x):
    return 1 / (1 + exp(-double_max(double_min(x, 20.0), -20.0)))


cdef class FM:
    """Factorization Machine online learner."""

    cdef unsigned int N
    cdef unsigned int k
    cdef double alpha
    cdef double w0
    cdef double c0
    cdef double[:] w
    cdef double[:] c
    cdef double[:] V

    def __init__(self, unsigned int N, unsigned int dim=4, double alpha=0.01):
        cdef int i

        random.seed(2014)
        self.N = N       # # of features
        self.k = dim
        self.alpha = alpha      # learning rate

        # initialize weights, factorized interactions, and counts
        self.w0 = 0.
        self.c0 = 0.
        self.w = np.zeros((self.N,), dtype=np.float64)
        self.c = np.zeros((self.N,), dtype=np.float64)
        self.V = np.zeros((self.N * self.k,), dtype=np.float64)
        for i in range(self.N * self.k):
            self.V[i] = (random.random() - .5) * .001

    def get_x(self, xs):
        """Apply hashing trick to a dictionary containing feature names and values.

        Args:
            xs - a list of "idx:value"

        Returns:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
        """
        idx = []
        val = []

        for item in xs:
            i, x = item.split(':')
            idx.append(int(i))
            val.append(float(x))

        return idx, val

    def predict(self, list idx, list val):
        """Predict for features.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features

        Returns:
            p - a prediction for input features
        """
        cdef int i
        cdef int k
        cdef double x
        cdef double p
        cdef double wx
        cdef double[:] vx
        cdef double[:] v2x2

        wx = 0.
        vx = np.zeros((self.k,), dtype=np.float64)
        v2x2 = np.zeros((self.k,), dtype=np.float64)
        for i, x in izip(idx, val):
            wx += self.w[i] * x
            for k in range(self.k):
                vx[k] += self.V[i * self.k + k] * x
                v2x2[k] += (self.V[i * self.k + k] ** 2) * (x ** 2)

        p = self.w0 + wx
        for k in range(self.k):
            p += .5 * (vx[k] ** 2 - v2x2[k])

        return sigm(p)

    def update(self, list idx, list val, double p, double y):
        """Update the model.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
            p - prediction of the model
            y - true target value

        Returns:
            updated model weights and counts
        """
        cdef int i
        cdef int k
        cdef int f
        cdef double x
        cdef double e
        cdef double abs_e
        cdef double dl_dw
        cdef double[:] vx

        # calculate v_f * x in advance
        vx = np.zeros((self.k,), dtype=np.float64)
        for i, x in izip(idx, val):
            for k in range(self.k):
                vx[k] += self.V[i * self.k + k] * x

        # update w0, w, V, c0, and c
        e = p - y
        abs_e = fabs(e)
        self.c0 += abs_e

        self.w0 -= self.alpha / (sqrt(self.c0) + 1) * e
        for i, x in izip(idx, val):
            dl_dw = self.alpha / (sqrt(self.c[i]) + 1) * e * x
            self.w[i] -= dl_dw
            self.c[i] += abs_e
            for f in range(self.k):
                self.V[i * self.k + f] -= dl_dw * (vx[f] -
                                                   self.V[i * self.k + f] * x)
