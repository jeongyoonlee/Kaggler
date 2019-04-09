# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

cimport cython
from libc.math cimport sqrt, abs
from ..util cimport sigm
cimport numpy as np


np.import_array()


cdef class NN:
    """Neural Network with a single ReLU hidden layer online learner.

    Attributes:
        n (int): number of input units
        epoch (int): number of epochs
        h (int): number of hidden units
        a (double): initial learning rate
        l2 (double): L2 regularization parameter
        w0 (array of double): weights between the input and hidden layers
        w1 (array of double): weights between the hidden and output layers
        z (array of double): hidden units
        c (double): counter
        c1 (array of double): counters for hidden units
    """

    cdef unsigned int epoch # number of epochs
    cdef unsigned int n     # number of input units
    cdef unsigned int h     # number of hidden units
    cdef double a           # learning rate
    cdef double l2          # L2 regularization parameter
    cdef double[:] w0       # weights between the input and hidden layers
    cdef double[:] w1       # weights between the hidden and output layers
    cdef double[:] z        # hidden units
    cdef double c           # counter
    cdef double[:] c0       # counters for input units
    cdef double[:] c1       # counters for hidden units

    def __init__(self,
                 unsigned int n,
                 unsigned int epoch=10,
                 unsigned int h=10,
                 double a=0.01,
                 double l2=0.,
                 unsigned int seed=0):
        """Initialize the NN class object.

        Args:
            n (int): number of input units
            epoch (int): number of epochs
            h (int): number of the hidden units
            a (double): initial learning rate
            l2 (double): L2 regularization parameter
            seed (unsigned int): random seed
        """

        cdef int i

        rng = np.random.RandomState(seed)

        self.epoch = epoch
        self.n = n
        self.h = h

        self.a = a
        self.l2 = l2

        self.w1 = (rng.rand(self.h + 1) - .5) * 1e-6
        self.w0 = (rng.rand((self.n + 1) * self.h) - .5) * 1e-6

        # hidden units in the hidden layer
        self.z = np.zeros((self.h,), dtype=np.float64)

        # counters for biases and inputs
        self.c = 0.
        self.c1 = np.zeros((self.h,), dtype=np.float64)
        self.c0 = np.zeros((self.n,), dtype=np.float64)

    def __repr__(self):
        return ('NN(n={}, epoch={}, h={}, a={}, l2={})').format(
            self.n, self.epoch, self.h, self.a, self.l2
        )

    def read_sparse(self, path):
        """Read a libsvm format sparse file line by line.

        Args:
            path (str): a file path to the libsvm format sparse file

        Yields:
            idx (list of int): a list of index of non-zero features
            val (list of double): a list of values of non-zero features
            y (int): target value
        """
        for line in open(path):
            xs = line.rstrip().split(' ')

            y = int(xs[0])
            idx = []
            val = []
            for item in xs[1:]:
                i, v = item.split(':')
                idx.append(int(i) % self.n)
                val.append(float(v))

            yield zip(idx, val), y

    cpdef fit(self, X, y):
        """Update the model with a sparse input feature matrix and its targets.

        Args:
            X (scipy.sparse.csr_matrix): a list of (index, value) of non-zero features
            y (numpy.array): targets

        Returns:
            updated model weights and counts
        """
        cdef int row

        cdef int[:] indices = X.indices
        cdef int[:] data = X.data
        cdef int[:] indptr = X.indptr

        for epoch in range(self.epoch):
            for row in range(X.shape[0]):
                x = zip(
                    indices[indptr[row] : indptr[row + 1]], 
                    data[indptr[row] : indptr[row + 1]],
                )
                self.update_one(x, self.predict_one(x) - y[row])
        return self

    def predict(self, X):
        """Predict for a sparse matrix X.

        Args:
            X (scipy.sparse.csr_matrix): a sparse matrix for input features

        Returns:
            p (numpy.array): predictions for input features
        """

        p = np.zeros((X.shape[0], ), dtype=np.float64)
        for row in range(X.shape[0]):
            p[row] = self.predict_one(zip(X[row].indices, X[row].data))

        return p

    def predict_one(self, list x):
        """Predict for features.

        Args:
            x (list of tuple): a list of (index, value) of non-zero features

        Returns:
            p (double): a prediction for input features
        """
        cdef double p
        cdef int j
        cdef int i
        cdef double v

        # starting with the bias in the hidden layer
        p = self.w1[self.h]

        # calculating and adding values of hidden units
        for j in range(self.h):
            # starting with the bias in the input layer
            self.z[j] = self.w0[self.n * self.h + j]

            # calculating and adding values of input units
            for i, v in x:
                self.z[j] += self.w0[i * self.h + j] * v

            # apply the ReLU activation function to the hidden unit
            self.z[j] = self.z[j] if self.z[j] > 0. else 0.

            p += self.w1[j] * self.z[j]

        # apply the sigmoid activation function to the output unit
        return sigm(p)

    def update_one(self, list x, double e):
        """Update the model with one observation.

        Args:
            x (list of tuple): a list of (index, value) of non-zero features
            e (double): error between the prediction of the model and target

        Returns:
            updated model weights and counts
        """
        cdef int j
        cdef int i
        cdef double dl_dy
        cdef double dl_dz
        cdef double dl_dw1
        cdef double dl_dw0
        cdef double v

        dl_dy = e      # dl/dy * (initial learning rate)

        # starting with the bias in the hidden layer
        self.w1[self.h] -= (dl_dy + self.l2 * self.w1[self.h]) * self.a / (sqrt(self.c) + 1)
        for j in range(self.h):
            # update weights related to non-zero hidden units
            if self.z[j] == 0.:
                continue

            # update weights between the hidden units and output
            # dl/dw1 = dl/dy * dy/dw1 = dl/dy * z
            dl_dw1 = dl_dy * self.z[j]
            self.w1[j] -= (dl_dw1 + self.l2 * self.w1[j]) * self.a / (sqrt(self.c1[j]) + 1)

            # starting with the bias in the input layer
            # dl/dz = dl/dy * dy/dz = dl/dy * w1
            dl_dz = dl_dy * self.w1[j]
            self.w0[self.n * self.h + j] -= (dl_dz +
                                             self.l2 * self.w0[self.n * self.h + j]) * self.a / (sqrt(self.c1[j]) + 1)
            # update weights related to non-zero input units
            for i, v in x:
                # update weights between the hidden unit j and input i
                # dl/dw0 = dl/dz * dz/dw0 = dl/dz * v
                dl_dw0 = dl_dz * v
                self.w0[i * self.h + j] -= (dl_dw0 +
                                            self.l2 * self.w0[i * self.h + j]) * self.a / (sqrt(self.c0[i]) + 1)

                # update counter for the input i
                self.c0[i] += dl_dw0 * dl_dw0

            # update counter for the hidden unit j
            self.c1[j] += dl_dw1 * dl_dw1

        # update overall counter
        self.c += dl_dy * dl_dy
