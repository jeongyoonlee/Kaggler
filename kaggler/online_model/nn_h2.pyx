from __future__ import division
import numpy as np
import random

cimport cython
from libc.math cimport sqrt, fabs
from ..util cimport sigm
cimport numpy as np


np.import_array()


cdef class NN_H2:
    """Neural Network with 2 ReLU hidden layers online learner."""

    cdef unsigned int n     # number of input units
    cdef unsigned int h1    # number of the 1st level hidden units
    cdef unsigned int h2    # number of the 2nd level hidden units
    cdef double a           # learning rate
    cdef double l1          # L1 regularization parameter
    cdef double l2          # L2 regularization parameter
    cdef double[:] w0       # weights between the input and 1st hidden layers
    cdef double[:] w1       # weights between the 1st and 2nd hidden layers
    cdef double[:] w2       # weights between the 2nd hidden and output layers
    cdef double[:] z1       # 1st level hidden units
    cdef double[:] z2       # 2nd level hidden units
    cdef double c           # counter
    cdef double[:] c0       # counters for input units
    cdef double[:] c1       # counters for 1st level hidden units
    cdef double[:] c2       # counters for 2nd level hidden units

    def __init__(self,
                 unsigned int n,
                 unsigned int h1=128,
                 unsigned int h2=256,
                 double a=0.01,
                 double l1=0.,
                 double l2=0.):
        cdef int i

        random.seed(2014)
        self.n = n
        self.h1 = h1
        self.h2 = h2

        self.a = a
        self.l1 = l1
        self.l2 = l2

        # weights between the output and 2nd hidden layer
        self.w2 = (np.random.rand(self.h2 + 1) - .5) * 1e-7

        # weights between the 2nd hidden layer and 1st hidden layer
        self.w1 = (np.random.rand((self.h1 + 1) * self.h2) - .5) * 1e-7

        # weights between the 1st hidden layer and inputs
        self.w0 = (np.random.rand((self.n + 1) * self.h1) - .5) * 1e-7

        # hidden units in the 2nd hidden layer
        self.z2 = np.zeros((self.h2,), dtype=np.float64)

        # hidden units in the 1st hidden layer
        self.z1 = np.zeros((self.h1,), dtype=np.float64)

        # counters for the hidden units and inputs
        self.c = 0.
        self.c2 = np.zeros((self.h2,), dtype=np.float64)
        self.c1 = np.zeros((self.h1,), dtype=np.float64)
        self.c0 = np.zeros((self.n,), dtype=np.float64)

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
                i, v = item.split(':')
                idx.append(fabs(hash(i)) % self.n)
                val.append(float(v))

            yield zip(idx, val), y

    def predict(self, list x):
        """Predict for features.

        Args:
            x - a list of (index, value) of non-zero features

        Returns:
            p - a prediction for input features
        """
        cdef double p
        cdef int k
        cdef int j
        cdef int i
        cdef double v

        # starting from the bias in the 2nd hidden layer
        p = self.w2[self.h2]

        # calculating and adding values of 2nd level hidden units
        for k in range(self.h2):
            # staring with the bias in the 1st hidden layer
            self.z2[k] = self.w1[self.h1 * self.h2 + k]

            # calculating and adding values of 1st level hidden units
            for j in range(self.h1):
                # starting with the bias in the input layer
                self.z1[j] = self.w0[self.n * self.h1 + j]

                # calculating and adding values of input units
                for i, v in x:
                    self.z1[j] += self.w0[i * self.h1 + j] * v

                # apply the ReLU activation function to the first level hidden unit
                self.z1[j] = self.z1[j] if self.z1[j] > 0. else 0.

                self.z2[k] += self.w1[j * self.h2 + k] * self.z1[j]

            # apply the ReLU activation function to the 2nd level hidden unit
            self.z2[k] = self.z2[k] if self.z2[k] > 0. else 0.

            p += self.w2[k] * self.z2[k]

        # apply the sigmoid activation function to the output unit
        return sigm(p)

    def update(self, list x, double e):
        """Update the model.

        Args:
            x - a list of (index, value) of non-zero features
            e - error between the prediction of the model and target

        Returns:
            updated model weights and counts
        """
        cdef int k
        cdef int j
        cdef int i
        cdef double abs_e
        cdef double dl_dy
        cdef double dl_dz1
        cdef double dl_dz2
        cdef double v

        # XXX: assuming predict() was called right before with the same idx and
        # val inputs.  Otherwise self.z will be incorrect for updates.
        abs_e = fabs(e)
        dl_dy = e * self.a      # dl/dy * (learning rate)

        # starting with the bias in the 2nd hidden layer
        self.w2[self.h2] -= dl_dy / (sqrt(self.c) + 1) + self.l2 * self.w2[self.h2]
        for k in range(self.h2):
            # update weights related to non-zero 2nd level hidden units
            if self.z2[k] == 0.:
                continue

            # update weights between the 2nd hidden units and output
            # dl/dw2 = dl/dy * dy/dw2 = dl/dy * z2
            self.w2[k] -= (dl_dy / (sqrt(self.c2[k]) + 1) * self.z2[k] +
                           self.l2 * self.w2[k])

            # starting with the bias in the 1st hidden layer
            # dl/dz2 = dl/dy * dy/dz2 = dl/dy * w2
            dl_dz2 = dl_dy * self.w2[k]
            self.w1[self.h1 * self.h2 + k] -= (dl_dz2 / (sqrt(self.c2[k]) + 1) +
                                               self.l2 * self.w1[self.h1 * self.h2 + k])
            for j in range(self.h1):
                # update weights realted to non-zero hidden units
                if self.z1[j] == 0.:
                    continue

                # update weights between the hidden units and output
                # dl/dw1 = dl/dz2 * dz2/dw1 = dl/dz2 * z1
                self.w1[j * self.h2 + k] -= (dl_dz2 / (sqrt(self.c1[j]) + 1) * self.z1[j] +
                                             self.l2 * self.w1[j])

                # starting with the bias in the input layer
                # dl/dz1 = dl/dz2 * dz2/dz1 = dl/dz2 * w1
                dl_dz1 = dl_dz2 * self.w1[j * self.h2 + k]
                self.w0[self.n * self.h1 + j] -= (dl_dz1 / (sqrt(self.c1[j]) + 1) +
                                                    self.l2 * self.w0[self.n * self.h1 + j])
                # update weights related to non-zero input units
                for i, v in x:
                    # update weights between the hidden unit j and input i
                    # dl/dw1 = dl/dz * dz/dw1 = dl/dz * v
                    self.w0[i * self.h1 + j] -= (dl_dz1 / (sqrt(self.c0[i]) + 1) * v +
                                                 self.l2 * self.w0[i * self.h1 + j])

                    # update counter for the input i
                    self.c0[i] += abs_e

                # update counter for the 1st level hidden unit j
                self.c1[j] += abs_e

            # update counter for the 2nd level hidden unit k
            self.c2[k] += abs_e

        # update overall counter
        self.c += abs_e
