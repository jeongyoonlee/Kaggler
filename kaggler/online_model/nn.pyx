from __future__ import division
from itertools import izip
import numpy as np
import random

cimport cython
from libc.math cimport sqrt, fabs
from ..util cimport sigm
cimport numpy as np


np.import_array()


cdef class NN:
    """Neural Network with a single ReLU hidden layer online learner."""

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
    cdef bint interaction   # use interaction between features

    def __init__(self,
                 unsigned int n,
                 unsigned int h=10,
                 double a=0.01,
                 double l2=0.,
                 bint interaction=True):
        cdef int i

        random.seed(2014)
        self.n = n
        self.h = h

        self.a = a
        self.l2 = l2

        self.w1 = (np.random.rand(self.h + 1) - .5) * 1e-6
        self.w0 = (np.random.rand((self.n + 1) * self.h) - .5) * 1e-6

        # hidden units in the hidden layer
        self.z = np.zeros((self.h,), dtype=np.float64)

        # counters for biases and inputs
        self.c = 0.
        self.c1 = np.zeros((self.h,), dtype=np.float64)
        self.c0 = np.zeros((self.n,), dtype=np.float64)

        # feature interaction
        self.interaction = interaction

    def get_x(self, xs):
        """Apply hashing trick to a dictionary of {feature name: value}.

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
        cdef double p
        cdef int j
        cdef int i
        cdef double x

        # starting with the bias in the hidden layer
        p = self.w1[self.h]

        # calculating and adding values of hidden units
        for j in range(self.h):
            # starting with the bias in the input layer
            self.z[j] = self.w0[self.n * self.h + j]

            # calculating and adding values of input units
            for i, x in izip(idx, val):
                self.z[j] += self.w0[i * self.h + j] * x

            # apply the ReLU activation function to the hidden unit
            self.z[j] = self.z[j] if self.z[j] > 0. else 0.

            p += self.w1[j] * self.z[j]

        # apply the sigmoid activation function to the output unit
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
        cdef int j
        cdef int i
        cdef double abs_e
        cdef double dl_dy
        cdef double dl_dz
        cdef double x

        # XXX: assuming predict() was called right before with the same idx and
        # val inputs.  Otherwise self.z will be incorrect for updates.
        abs_e = fabs(e)
        dl_dy = e * self.a      # dl/dy * (learning rate)

        # starting with the bias in the hidden layer
        self.w1[self.h] -= dl_dy / (sqrt(self.c) + 1) + self.l2 * self.w1[self.h]
        for j in range(self.h):
            # update weights related to non-zero hidden units
            if self.z[j] == 0.:
                continue

            # update weights between the hidden units and output
            # dl/dw1 = dl/dy * dy/dw1 = dl/dy * z
            self.w1[j] -= (dl_dy / (sqrt(self.c1[j]) + 1) * self.z[j] +
                           self.l2 * self.w1[j])

            # starting with the bias in the input layer
            # dl/dz = dl/dy * dy/dz = dl/dy * w1
            dl_dz = dl_dy * self.w1[j]
            self.w0[self.n * self.h + j] -= (dl_dz / (sqrt(self.c1[j]) + 1) +
                                             self.l2 * self.w0[self.n * self.h + j])
            # update weights related to non-zero input units
            for i, x in izip(idx, val):
                # update weights between the hidden unit j and input i
                # dl/dw0 = dl/dz * dz/dw0 = dl/dz * x
                self.w0[i * self.h + j] -= (dl_dz / (sqrt(self.c0[i]) + 1) * x +
                                            self.l2 * self.w0[i * self.h + j])

                # update counter for the input i
                self.c0[i] += abs_e

            # update counter for the hidden unit j
            self.c1[j] += abs_e

        # update overall counter
        self.c += abs_e
