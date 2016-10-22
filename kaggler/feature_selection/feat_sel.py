from scipy import sparse
from sklearn import base
import numpy as np


"""
Code below was originally written by Baris Umog (https://www.kaggle.com/barisumog).
"""
class DropInactive(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, lowest=25):
        self.lowest = lowest

    def fit(self, X, y=None):
        x = (X > 0.0).astype(bool)
        s = np.array(x.sum(axis=0)).flatten()
        self.mask = (s >= self.lowest)
        return self

    def transform(self, X):
        print(self.mask.sum())
        return X[:, self.mask]


class DropLowInfo(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, margin=0.02, weighted=True):
        self.margin = margin
        self.weighted = weighted

    def fit(self, X, y=None):
        mean = y.mean()
        lower = mean - self.margin
        upper = mean + self.margin
        ys = sparse.csc_matrix(y[:, np.newaxis])
        if self.weighted:
            x = X.multiply(ys).sum(axis=0)
            x = x / X.sum(axis=0)
        else:
            x = (X > 0)
            s = x.sum(axis=0)
            x = x.multiply(ys).sum(axis=0) / s
        x = np.array(x).flatten().astype('f4')
        mask1 = (x < lower)
        mask2 = (x > upper)
        self.mask = (mask1 + mask2).astype(bool)
        return self

    def transform(self, X):
        print(self.mask.sum())
        return X[:, self.mask]
