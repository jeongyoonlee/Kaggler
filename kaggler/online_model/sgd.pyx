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


cdef class SGD:
    """Simple online learner using a hasing trick.

    Attributes:
        n (int): number of features after hashing trick
        a (double): initial learning rate
        l1 (double): L1 regularization parameter
        l2 (double): L2 regularization parameter
        w (array of double): feature weights
        c (array of double): counters for weights
        interaction (boolean): whether to use 2nd order interaction or not
    """
    cdef unsigned int n
    cdef double a
    cdef double l1
    cdef double l2
    cdef double[:] w
    cdef double[:] c
    cdef bint interaction

    def __init__(self,
                 unsigned int n=2**20,
                 double a=0.01,
                 double l1=0.0,
                 double l2=0.0,
                 bint interaction=True):
        """Initialize the SGD class object.

        Args:
            n (int): number of features after hashing trick
            a (double): initial learning rate
            l1 (double): L1 regularization parameter
            l2 (double): L2 regularization parameter
            w (array of double): feature weights
            c (array of double): counters for weights
            interaction (boolean): whether to use 2nd order interaction or not
        """

        self.n = n      # # of features
        self.a = a      # learning rate
        self.l1 = l1
        self.l2 = l2

        # initialize weights and counts
        self.w = np.zeros((self.n,), dtype=np.float64)
        self.c = np.zeros((self.n,), dtype=np.float64)
        self.interaction = interaction

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

    def predict(self, list x):
        """Predict for features.

        Args:
            x (list of int): a list of index of non-zero features

        Returns:
            p (double): a prediction for input features
        """
        cdef int i
        cdef double wTx

        wTx = 0.
        for i in self._indices(x):
            wTx += self.w[i]

        return sigm(wTx)

    def update(self, list x, double e):
        """Update the model.

        Args:
            x (list of int): a list of index of non-zero features
            e (double): error between the prediction of the model and target

        Returns:
            updates model weights and counts
        """
        cdef int i
        cdef double g2

        g2 = e * e
        for i in self._indices(x):
            self.w[i] -= (e +
                          (self.l1 if self.w[i] >= 0. else -self.l1) +
                          self.l2 * self.w[i]) * self.a / (sqrt(self.c[i]) + 1)
            self.c[i] += g2
