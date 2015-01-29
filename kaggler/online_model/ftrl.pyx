from __future__ import division
import numpy as np

cimport cython
from libc.math cimport exp, sqrt
cimport numpy as np


np.import_array()

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b


cdef double sigm(double x):
    return 1 / (1 + exp(-double_max(double_min(x, 20.0), -20.0)))


cdef class FTRL:
    """FTRL online learner with the hasing trick using liblinear format data.
    
    inspired by Kaggle user tinrtgu's code at http://goo.gl/K8hQBx
    original FTRL paper is available at http://goo.gl/iqIaH0
    """

    cdef double alpha      # learning rate
    cdef double beta
    cdef double l1
    cdef double l2
    cdef unsigned int N              # # of features
    cdef bint interaction
    cdef double[:] w
    cdef double[:] c
    cdef double[:] z
    #cdef dict z

    def __init__(self,
                 double alpha=0.01,
                 double beta=1.,
                 double l1=1.,
                 double l2=1.,
                 unsigned int N=2**20,
                 bint interaction=True):
        self.alpha = alpha      # learning rate
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.N = N              # # of features
        self.interaction = interaction

        # initialize weights and counts
        self.w = np.zeros((self.N,), dtype=np.float64)
        self.c = np.zeros((self.N,), dtype=np.float64)
        self.z = np.zeros((self.N,), dtype=np.float64)
        #self.z = {}

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
                    yield abs(hash('{}_{}'.format(x[i], x[j]))) % self.N

    def get_x(self, list xs):
        """Apply hashing trick to a dictionary of {feature name: value}.

        Args:
            xs - a list of "idx:value"

        Returns:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
        """
        x = []
        for item in xs:
            index, _ = item.split(':')
            x.append(abs(hash(index)) % self.N)

        return x

    def update(self, list x, double p, double y):
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
        cdef double g
        cdef double g2
        cdef double s

        g = p - y
        g2 = g * g
        for i in self._indices(x):
            s = (sqrt(self.c[i] + g2) - sqrt(self.c[i])) / self.alpha
            self.w[i] += g - s * self.z[i]
            self.c[i] += g2

    def predict(self, list x):
        """Predict for features.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features

        Returns:
            a prediction for input features
        """
        cdef int i
        cdef double sign
        cdef double wTx
        #cdef dict z

        #z = {}
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if self.w[i] < 0 else 1.
            if sign * self.w[i] <= self.l1:
                self.z[i] = 0.
            else:
                self.z[i] = (sign * self.l1 - self.w[i]) / \
                            ((self.beta + sqrt(self.c[i])) / self.alpha + self.l2)

            wTx += self.z[i]

        #self.z = z

        return sigm(wTx)
