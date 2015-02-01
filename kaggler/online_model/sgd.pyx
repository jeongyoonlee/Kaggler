from __future__ import division
from itertools import izip
import numpy as np
import random

cimport cython
from libc.math cimport sqrt, fabs
from ..util cimport sigm
cimport numpy as np


np.import_array()


cdef class SGD:
    cdef unsigned int n
    cdef double a
    cdef double[:] w
    cdef double[:] c

    """Simple online learner using a hasing trick."""

    def __init__(self, unsigned int n, double a=0.01):
        self.n = n       # # of features
        self.a = a      # learning rate

        # initialize weights and counts
        self.w = np.zeros((self.n,), dtype=np.float64)
        self.c = np.zeros((self.n,), dtype=np.float64)

    def get_x(self, xs):
        """Apply hashing trick to a dictionary of {feature name: value}.

        Args:
            xs - a list of "idx:value"

        Returns:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
        """
        idx = [0] # 0 is the index of the bias term
        val = [1.] # 1 is the value of the bias term

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
            a prediction for input features
        """
        cdef int i
        cdef double x
        cdef double wTx

        wTx = 0.
        for i, x in izip(idx, val):
            wTx += self.w[i] * x

        return sigm(wTx)

    def update(self, list idx, list val, double p, double y):
        """Update the model.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
            p - prediction of the model
            y - true target value

        Returns:
            updates model weights and counts
        """
        cdef int i
        cdef double x
        cdef double e

        e = p - y
        for i, x in izip(idx, val):
            self.w[i] -= e * x * self.a / (sqrt(self.c[i]) + 1)
            self.c[i] += fabs(e)
