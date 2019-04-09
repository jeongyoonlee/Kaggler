# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import sparse
from keras import backend as K

import logging
import numpy as np
import tensorflow as tf
import os

cimport cython
from libc.math cimport exp, log
cimport numpy as np


np.import_array()
logger = logging.getLogger('Kaggler')


cdef double sigm(double x):
    """Bounded sigmoid function."""
    return 1 / (1 + exp(-fmax(fmin(x, 20.0), -20.0)))


def get_downsampled_index(n, rate=0.):
    """Return the index that downsamples a vector x by the rate."""

    return np.random.choice(range(n), int(n * rate), replace=False)


def get_downsampled_index0(x, rate=0., threshold=0.):
    """Return the index that downsamples 0s of a vector x by the rate."""

    idx1 = np.where(x > threshold)[0]
    idx0 = np.where(x <= threshold)[0]
    idx0_down = np.random.choice(idx0, int(len(idx0) * rate), replace=False)

    idx = list(idx0_down) + list(idx1)
    np.random.shuffle(idx)

    return idx


def set_column_width(X, n_col):
    """Set the column width of a matrix X to n_col."""

    if X.shape[1] < n_col:
        if sparse.issparse(X):
            X = sparse.hstack((X, np.zeros((X.shape[0], n_col - X.shape[1]))))
            X = X.tocsr()
        else:
            X = np.hstack((X, np.zeros((X.shape[0], n_col - X.shape[1]))))

    elif X.shape[1] > n_col:
        if sparse.issparse(X):
            X = X.tocsc()[:, :-(X.shape[1] - n_col)]
            X = X.tocsr()
        else:
            X = X[:, :-(X.shape[1] - n_col)]

    return X


def rank(x):
    """Rank a vector x.  Ties will be averaged."""

    unique, idx_inverse = np.unique(x, return_inverse=True)

    unique_rank_sum = np.zeros_like(unique)
    unique_rank_count = np.zeros_like(unique)

    np.add.at(unique_rank_sum, idx_inverse, x.argsort().argsort())
    np.add.at(unique_rank_count, idx_inverse, 1)

    unique_rank_mean = unique_rank_sum.astype(np.float) / unique_rank_count

    return unique_rank_mean[idx_inverse]


def set_min_max(x, lb, ub):
    x[x < lb] = lb
    x[x > ub] = ub

    return x


def point(rank, n_team, n_teammate=1, t=0):
    """Calculate Kaggle points to earn after a competition.

    Args:
        rank (int): final ranking in the private leaderboard.
        n_team (int): the number of teams participated in the competition.
        n_teammate (int): the number of team members in my team.
        t (int): the number of days since the competition ends.

    Returns:
        returns Kaggle points to earn after a compeittion.
    """
    return (1e5 / np.sqrt(n_teammate) * (rank ** -.75) *
            np.log10(1 + np.log10(n_team)) * np.exp(-t / 500))


def limit_mem(gpu=0):
    gpu = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
    logger.info('using GPU #{}'.format(gpu))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
