# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from scipy import sparse
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from statsmodels.distributions.empirical_distribution import ECDF

import logging
import numpy as np
import pandas as pd

cimport cython
from libc.math cimport exp, log
cimport numpy as np


NAN_STR = '__Kaggler_NaN__'
np.import_array()

cdef inline double fmax(double a, double b): return a if a >= b else b
cdef inline double fmin(double a, double b): return a if a <= b else b


cdef double sigm(double x):
    """Bounded sigmoid function."""
    return 1 / (1 + exp(-fmax(fmin(x, 20.0), -20.0)))


def logloss(double p, double y):
    p = fmax(fmin(p, 1. -1e-15), 1e-15)
    return -log(p) if y == 1. else -log(1. - p)


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


def normalize_numerical_features(df, n=None):
    """Normalize numerical Pandas columns.
    
    Args:
        df (pandas.DataFrame) : numerical columns
        n: number of observations to use for deriving probability distribution
           of the feature.  Observations beyond first n observations will be
           normalized based on the probability distribution found from the
           first n observations. 

    Returns:
        df (pandas.DataFrame): normalized numerical Pandas columns
    """

    for col in df.columns:
        df[col] = normalize_numerical_feature(df[col], n)
        
    return df


def normalize_numerical_feature(feature, n=None):
    """Normalize a numerical Pandas column.
    
    Args:
        feature: feature vector to normalize.
        n: number of observations to use for deriving probability distribution
           of the feature.  Observations beyond first n observations will be
           normalized based on the probability distribution found from the
           first n observations. 

    Returns:
        A normalized feature vector.
    """

    if not n:
        n = len(feature)

    # add one to the numerator and denominator to avoid +-inf.
    ecdf = ECDF(feature[:n])

    return norm.ppf(ecdf(feature) * .998 + .001)


def get_label_encoder(feature, min_obs=10, nan_as_var=False):
    """Return a mapping from values of features to integer labels.

    Args:
        feature (pandas.Series): categorical feature column to encode
        min_obs (int): minimum number of observation to assign a label
        nan_as_var (bool): whether to create a dummy variable for NaN or not

    Returns:
        label_encoder (dict): mapping from values of features to integer labels
    """
    label_count = {}
    for label in feature:
        try:
            label_count[label] += 1
        except KeyError:
            label_count[label] = 1

    label_encoder = {}
    label_index = 1
    for label in label_count.keys():
        if (not nan_as_var) and label == NAN_STR:
            label_encoder[label] = -1
        elif label_count[label] >= min_obs:
            label_encoder[label] = label_index
            label_index += 1

    return label_encoder


def encode_categorical_features(df, min_obs=10, n=None, nan_as_var=False):
    """Encode Pandas columns into sparse matrix with one-hot-encoding.

    Args:
        df (pandas.DataFrame): categorical feature columns to encode
        min_obs (int): minimum number of observation to create a dummy variable
        n (int): number of observation to be used to create dummy variables
        nan_as_var (bool): whether to create a dummy variable for NaN or not

    Returns:
        X (scipy.sparse.coo_matrix): sparse matrix encoding a categorical
                                     variable into dummy variables
    """

    n_feature = 0
    for i, col in enumerate(df.columns):
        X_col = encode_categorical_feature(df[col], min_obs, n, nan_as_var)
        if X_col is not None:
            if i == 0:
                X = X_col
            else:
                X = sparse.hstack((X, X_col))

        logging.debug('{} --> {} features'.format(col, X.shape[1] - n_feature))
        n_feature = X.shape[1]

    return X


def encode_categorical_feature(feature, min_obs=10, n=None, nan_as_var=False):
    """Encode a Pandas column into sparse matrix with one-hot-encoding.

    Args:
        feature (pandas.Series): categorical feature column to encode
        min_obs (int): minimum number of observation to create a dummy variable
        n (int): number of observation to be used to create dummy variables
        nan_as_var (bool): whether to create a dummy variable for NaN or not

    Returns:
        X (scipy.sparse.coo_matrix): sparse matrix encoding a categorical
                                     variable into dummy variables.
    """

    # impute missing values with a custom string so that we can count number of
    # NaN
    feature.fillna(NAN_STR, inplace=True)

    n_obs = len(feature)
    if not n:
        n = n_obs

    label_encoder = get_label_encoder(feature[:n], min_obs, nan_as_var)

    labels = feature.apply(lambda x: label_encoder.get(x, 0))
    labels.index = range(len(labels))
    if labels[labels == 0].count() >= min_obs:
        i = labels.index[labels >= 0].values
        j = labels[labels >= 0].values
    else:
        i = labels.index[labels > 0].values
        j = labels[labels > 0].values - 1

    if len(i) > 0:
        return sparse.coo_matrix((np.ones_like(i), (i, j)),
                                 shape=(n_obs, j.max() + 1))
    else:
        return None


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
