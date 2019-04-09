# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: linetrace=False
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np


cdef extern from "murmurhash/MurmurHash3.h":
    void MurmurHash3_x86_32(void *key, int len, np.uint32_t seed, void *out)

cdef int murmurhash3_int_s32(int key, unsigned int seed):
    """Compute the 32bit murmurhash3 of a int key at seed."""
    cdef int out
    MurmurHash3_x86_32(&key, sizeof(int), seed, &out)
    return out


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
    cdef int epoch
    cdef int n
    cdef bint interaction
    cdef double[:] w
    cdef double[:] c
    cdef double[:] z

    def __init__(self,
                 double a=0.01,
                 double b=1.,
                 double l1=1.,
                 double l2=1.,
                 int n=2**20,
                 int epoch=1,
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

    cdef list _indices(self, int[:] x):
        cdef int x_len = x.shape[0]
        cdef int index
        cdef int i
        cdef int j
        cdef list indices = []
        indices.append(self.n)

        for i in range(x_len):
            index = x[i]
            indices.append(index % self.n)

        if self.interaction:
            for i in range(x_len - 1):
                for j in range(i + 1, x_len):
                    index = abs(murmurhash3_int_s32(x[i] * x[j], seed=0))
                    indices.append(index % self.n)
        return indices

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
                x.append(int(index))

            yield x, y

    def fit(self, X, y):
        """Update the model with a sparse input feature matrix and its targets.

        Args:
            X (scipy.sparse.csr_matrix): a list of (index, value) of non-zero features
            y (numpy.array): targets

        Returns:
            updated model weights and counts
        """
        if y.dtype != np.float64:
            y = y.astype(np.float64)
        self._fit(X, y)
        return self

    cdef void _fit(self, X, double[:] y):
        """Update the model with a sparse input feature matrix and its targets.

        Args:
            X (scipy.sparse.csr_matrix): a list of (index, value) of non-zero features
            y (numpy.array): targets

        Returns:
            updated model weights and counts
        """
        cdef int row
        cdef int row_num = X.shape[0]

        cdef int[:] x
        cdef int[:] indices = X.indices
        cdef int[:] indptr = X.indptr

        for epoch in range(self.epoch):
            for row in range(row_num):
                x = indices[indptr[row] : indptr[row + 1]]
                self._update_one(x, self._predict_one(x) - y[row])

    def predict(self, X):
        """Predict for a sparse matrix X.

        Args:
            X (scipy.sparse.csr_matrix): a sparse matrix for input features

        Returns:
            p (numpy.array): predictions for input features
        """
        return self._predict(X)

    cdef _predict(self, X):
        """Predict for a sparse matrix X.

        Args:
            X (scipy.sparse.csr_matrix): a sparse matrix for input features

        Returns:
            p (numpy.array): predictions for input features
        """
        cdef int row
        cdef int row_num = X.shape[0]
        cdef int[:] x
        cdef int[:] indices = X.indices
        cdef int[:] indptr = X.indptr

        p = np.zeros((row_num, ), dtype=np.float64)
        for row in range(row_num):
            x = indices[indptr[row] : indptr[row + 1]]
            p[row] = self._predict_one(x)
        return p

    def update_one(self, x, e):
        x = np.array(x, dtype=int)
        self._update_one(x, e)

    cpdef void _update_one(self, int[:] x, double e):
        """Update the model.

        Args:
            x (list of int): a list of index of non-zero features
            e (double): error between prediction of the model and target

        Returns:
            updates model weights and counts
        """
        cdef int i
        cdef int j
        cdef double e2
        cdef double s
        cdef list indices = self._indices(x)
        cdef int indices_num = len(indices)

        e2 = e * e
        for j in range(indices_num):
            i = indices[j]
            s = (sqrt(self.c[i] + e2) - sqrt(self.c[i])) / self.a
            self.w[i] += e - s * self.z[i]
            self.c[i] += e2

    def predict_one(self, x):
        x = np.array(x, dtype=int)
        return self._predict_one(x)

    cpdef double _predict_one(self, int[:] x):
        """Predict for features.

        Args:
            x (list of int): a list of index of non-zero features

        Returns:
            p (double): a prediction for input features
        """
        cdef int i
        cdef int j
        cdef double sign
        cdef double wTx
        cdef list indices = self._indices(x)
        cdef int indices_num = len(indices)

        wTx = 0.
        for j in range(indices_num):
            i = indices[j]
            sign = -1. if self.w[i] < 0 else 1.
            if sign * self.w[i] <= self.l1:
                self.z[i] = 0.
            else:
                self.z[i] = (sign * self.l1 - self.w[i]) / \
                            ((self.b + sqrt(self.c[i])) / self.a + self.l2)

            wTx += self.z[i]

        return sigm(wTx)
