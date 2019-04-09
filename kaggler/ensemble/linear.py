from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def netflix(es, ps, e0, l=.0001):
    """
    Combine predictions with the optimal weights to minimize RMSE.

    Args:
        es (list of float): RMSEs of predictions
        ps (list of np.array): predictions
        e0 (float): RMSE of all zero prediction
        l (float): lambda as in the ridge regression

    Returns:
        Ensemble prediction (np.array) and weights (np.array) for input predictions
    """
    m = len(es)
    n = len(ps[0])

    X = np.stack(ps).T
    pTy = .5 * (n * e0**2 + (X**2).sum(axis=0) - n * np.array(es)**2)

    w = np.linalg.pinv(X.T.dot(X) + l * n * np.eye(m)).dot(pTy)

    return X.dot(w), w
