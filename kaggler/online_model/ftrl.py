from __future__ import division
from math import sqrt

from ..util import sigm


class FTRL(object):
    """FTRL online learner with the hasing trick using liblinear format data.
    
    inspired by Kaggle user tinrtgu's code at http://goo.gl/K8hQBx
    original FTRL paper is available at http://goo.gl/iqIaH0
    """

    def __init__(self, alpha=0.01, beta=1., l1=1., l2=1., N=2**20,
                 interaction=True):
        self.alpha = alpha      # learning rate
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.N = N              # # of features
        self.interaction = interaction

        # initialize weights and counts
        self.w = [0.] * self.N
        self.c = [0.] * self.N
        self.z = {}

    def _indices(self, x):
        yield 0

        for index in x:
            yield index

        if self.interaction:
            l = len(x)
            x = sorted(x)
            for i in xrange(l):
                for j in xrange(i + 1, l):
                    yield abs(hash('{}_{}'.format(x[i], x[j]))) % self.N

    def get_x(self, xs):
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

    def update(self, x, yhat, y):
        """Update the model.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features
            yhat - prediction of the model
            y - true target value

        Returns:
            updates model weights and counts
        """
        g = yhat - y
        g2 = g * g
        for i in self._indices(x):
            s = (sqrt(self.c[i] + g2) - sqrt(self.c[i])) / self.alpha
            self.w[i] += g - s * self.z[i]
            self.c[i] += g2

    def predict(self, x):
        """Predict for features.

        Args:
            idx - a list of index of non-zero features
            val - a list of values of non-zero features

        Returns:
            a prediction for input features
        """
        z = {}
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if self.w[i] < 0 else 1.
            if sign * self.w[i] <= self.l1:
                z[i] = 0.
            else:
                z[i] = (sign * self.l1 - self.w[i]) / \
                        ((self.beta + sqrt(self.c[i])) / self.alpha + self.l2)

            wTx += z[i]

        self.z = z

        return sigm(wTx)
