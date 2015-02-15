# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from __future__ import division
import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np


np.import_array()


cdef class FTRL:
    """FTRL online learner with the hasing trick using liblinear format data.
    
    inspired by Kaggle user tinrtgu's code at http://goo.gl/K8hQBx
    original FTRL paper is available at http://goo.gl/iqIaH0

    Attributes:
        n (int): number of features after hashing trick
        a (double): alpha in the per-coordinate rate
        b (double): beta in the per-coordinate rate
        l1 (double): L1 regularization parameter
        l2 (double): L2 regularization parameter
        w (array of double): feature weights
        c (array of double): counters for weights
        z (array of double): lazy weights
        interaction (boolean): whether to use 2nd order interaction or not
    """

    cdef double a      # learning rate
    cdef double b
    cdef double l1
    cdef double l2
    cdef unsigned int n              # # of features
    cdef bint interaction
    cdef double[:] w
    cdef double[:] c
    cdef double[:] z

    def __init__(self,
                 double a=0.01,
                 double b=1.,
                 double l1=1.,
                 double l2=1.,
                 unsigned int n=2**20,
                 bint interaction=True):
        """Initialize the FTRL class object.

        Args:
            a (double): alpha in the per-coordinate rate
            b (double): beta in the per-coordinate rate
            l1 (double): L1 regularization parameter
            l2 (double): L2 regularization parameter
            n (int): number of features after hashing trick
            interaction (boolean): whether to use 2nd order interaction or not
        """

        self.a = a      # learning rate
        self.b = b
        self.l1 = l1
        self.l2 = l2
        self.n = n              # # of features
        self.interaction = interaction

        # initialize weights and counts
        self.w = np.zeros((self.n,), dtype=np.float64)
        self.c = np.zeros((self.n,), dtype=np.float64)
        self.z = np.zeros((self.n,), dtype=np.float64)

    def _indices(self, list x):
        cdef unsigned int index
        cdef int l
        cdef int i
        cdef int j

        yield 0

        for index in x:
            yield index

        if self.interaction:
            l = len(x)
            x = sorted(x)
            for i in xrange(l):
                for j in xrange(i + 1, l):
                    yield abs(hash('{}_{}'.format(x[i], x[j]))) % self.n

    def read_sparse(self, path):
        """Apply hashing trick to the libsvm format sparse file.

        Args:
            path (str): a file path to the libsvm format sparse file

        Yields:
            x (list of int): a list of index of non-zero features
            y (int): target value
        """
        for line in open(path):
            xs = line.rstrip().split(' ')

            y = int(xs[0])
            x = []
            for item in xs[1:]:
                index, _ = item.split(':')
                x.append(abs(hash(index)) % self.n)

            yield x, y

    def update(self, list x, double e):
        """Update the model.

        Args:
            x (list of int): a list of index of non-zero features
            e (double): error between prediction of the model and target

        Returns:
            updates model weights and counts
        """
        cdef int i
        cdef double e2
        cdef double s

        e2 = e * e
        for i in self._indices(x):
            s = (sqrt(self.c[i] + e2) - sqrt(self.c[i])) / self.a
            self.w[i] += e - s * self.z[i]
            self.c[i] += e2

    def predict(self, list x):
        """Predict for features.

        Args:
            x (list of int): a list of index of non-zero features

        Returns:
            p (double): a prediction for input features
        """
        cdef int i
        cdef double sign
        cdef double wTx

        wTx = 0.
        for i in self._indices(x):
            sign = -1. if self.w[i] < 0 else 1.
            if sign * self.w[i] <= self.l1:
                self.z[i] = 0.
            else:
                self.z[i] = (sign * self.l1 - self.w[i]) / \
                            ((self.b + sqrt(self.c[i])) / self.a + self.l2)

            wTx += self.z[i]

        return sigm(wTx)
