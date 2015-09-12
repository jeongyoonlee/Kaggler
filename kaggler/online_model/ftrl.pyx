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
        epoch (int): number of epochs
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
    cdef unsigned int epoch
    cdef unsigned int n
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
                 unsigned int epoch=1,
                 bint interaction=True):
        """Initialize the FTRL class object.

        Args:
            a (double): alpha in the per-coordinate rate
            b (double): beta in the per-coordinate rate
            l1 (double): L1 regularization parameter
            l2 (double): L2 regularization parameter
            n (int): number of features after hashing trick
            epoch (int): number of epochs
            interaction (boolean): whether to use 2nd order interaction or not
        """

        self.a = a
        self.b = b
        self.l1 = l1
        self.l2 = l2
        self.n = n
        self.epoch = epoch
        self.interaction = interaction

        # initialize weights and counts
        self.w = np.zeros((self.n + 1,), dtype=np.float64)
        self.c = np.zeros((self.n + 1,), dtype=np.float64)
        self.z = np.zeros((self.n + 1,), dtype=np.float64)

    def __repr__(self):
        return ('FTRL(a={}, b={}, l1={}, l2={}, n={}, epoch={}, interaction={})').format(
            self.a, self.b, self.l1, self.l2, self.n, self.epoch, self.interaction
        )

    def _indices(self, list x):
        cdef unsigned int index
        cdef int l
        cdef int i
        cdef int j

        # return the index of the bias term
        yield self.n

        for index in x:
            yield abs(hash(index)) % self.n

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
                x.append(index)

            yield x, y

    def fit(self, X, y):
        """Update the model with a sparse input feature matrix and its targets.

        Args:
            X (scipy.sparse.csr_matrix): a list of (index, value) of non-zero features
            y (numpy.array): targets

        Returns:
            updated model weights and counts
        """
        for epoch in range(self.epoch):
            for row in range(X.shape[0]):
                x = list(X[row].indices)
                self.update_one(x, self.predict_one(x) - y[row])

    def predict(self, X):
        """Predict for a sparse matrix X.

        Args:
            X (scipy.sparse.csr_matrix): a sparse matrix for input features

        Returns:
            p (numpy.array): predictions for input features
        """
        p = np.zeros((X.shape[0], ), dtype=np.float64)
        for row in range(X.shape[0]):
            p[row] = self.predict_one(list(X[row].indices))

        return p

    def update_one(self, list x, double e):
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

    def predict_one(self, list x):
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
