from scipy import sparse
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from statsmodels.distributions.empirical_distribution import ECDF

import numpy as np

cimport cython
from libc.math cimport exp, log
cimport numpy as np


np.import_array()

cdef inline double fmax(double a, double b): return a if a >= b else b
cdef inline double fmin(double a, double b): return a if a <= b else b


cdef double sigm(double x):
    """Bounded sigmoid function."""
    return 1 / (1 + exp(-fmax(fmin(x, 20.0), -20.0)))


cdef double logloss(double p, double y):
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


def is_number(s):
    """Check if a string is a number or not."""

    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_numerical_feature(feature):
    """Normalize the Pandas column based on cumulative distribution."""

    # add one to the numerator and denominator to avoid +-inf.
    p = (1 + rank(np.array(feature.astype(np.float64)))) / (1 + len(feature))
    return norm.ppf(p)


def normalize_numerical_feature2(feature, n=None):
    """Normalize the Pandas column based on cumulative distribution.
    
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


def get_label_encoder(feature, min_obs=10):
    label_count = {}
    for label in feature:
        try:
            label_count[label] += 1
        except KeyError:
            label_count[label] = 1

    label_encoder = {}
    label_index = 1
    for label in label_count.keys():
        if label_count[label] >= min_obs:
            label_encoder[label] = label_index
            label_index += 1

    return label_encoder


def encode_categorical_feature(feature, min_obs=10, n=None):
    """Encode the Pandas column into sparse matrix with one-hot-encoding."""

    if not n:
        n = len(feature)

    label_encoder = get_label_encoder(feature[:n], min_obs)
    labels = feature.apply(lambda x: label_encoder.get(x, 0))
    enc = OneHotEncoder()

    return enc.fit_transform(np.matrix(labels).reshape(len(labels), 1))


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
