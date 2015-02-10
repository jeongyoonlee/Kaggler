from __future__ import division
from itertools import izip
import numpy as np
import random

cimport cython
from libc.math cimport sqrt, fabs
from ..util cimport sigm
cimport numpy as np


np.import_array()


cdef class FM:
    """Factorization Machine online learner."""

    cdef unsigned int n
    cdef unsigned int k
    cdef double a
    cdef double w0
    cdef double c0
    cdef double[:] w
    cdef double[:] c
    cdef double[:] V

    def __init__(self, unsigned int n, unsigned int dim=4, double a=0.01):
        cdef int i

        random.seed(2014)
        self.n = n       # # of features
        self.k = dim
        self.a = a      # learning rate

        # initialize weights, factorized interactions, and counts
        self.w0 = 0.
        self.c0 = 0.
        self.w = np.zeros((self.n,), dtype=np.float64)
        self.c = np.zeros((self.n,), dtype=np.float64)
        self.V = (np.random.rand(self.n * self.k) - .5) * 1e-6

    def read_sparse(self, path):
        """Apply hashing trick to the libsvm format sparse file.

        Args:
            path - a file path to the libsvm format sparse file

        Returns:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
            y - target value
        """
        for line in open(path):
            xs = line.rstrip().split(' ')

            y = int(xs[0])
            idx = []
            val = []
            for item in xs[1:]:
                i, x = item.split(':')
                idx.append(fabs(hash(i)) % self.n)
                val.append(float(x))

            yield idx, val, y

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

    def update(self, list idx, list val, double e):
        """Update the model.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
            e - error between the prediction of the model and target

        Returns:
            updated model weights and counts
        """
        cdef int i
        cdef int k
        cdef int f
        cdef double x
        cdef double abs_e
        cdef double dl_dw
        cdef double[:] vx

        # calculate v_f * x in advance
        vx = np.zeros((self.k,), dtype=np.float64)
        for i, x in izip(idx, val):
            for k in range(self.k):
                vx[k] += self.V[i * self.k + k] * x

        # update w0, w, V, c0, and c
        abs_e = fabs(e)
        self.c0 += abs_e

        self.w0 -= self.a / (sqrt(self.c0) + 1) * e
        for i, x in izip(idx, val):
            dl_dw = self.a / (sqrt(self.c[i]) + 1) * e * x
            self.w[i] -= dl_dw
            self.c[i] += abs_e
            for f in range(self.k):
                self.V[i * self.k + f] -= dl_dw * (vx[f] -
                                                   self.V[i * self.k + f] * x)
