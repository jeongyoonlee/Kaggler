from __future__ import division
from scipy import sparse
from scipy.signal import butter, lfilter
from scipy.stats import norm
from sklearn import base
from statsmodels.distributions.empirical_distribution import ECDF
import logging
import numpy as np
import pandas as pd


NAN_INT = 7535805


class Normalizer(base.BaseEstimator):
    """Normalizer that transforms numerical columns into normal distribution.

    Attributes:
        ecdfs (list of empirical CDF): empirical CDFs for columns
    """

    def fit(self, X, y=None):
        self.ecdfs = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.ecdfs[col] = ECDF(X[:, col])

        return self

    def transform(self, X):
        """Normalize numerical columns.
        
        Args:
            X (numpy.array) : numerical columns to normalize

        Returns:
            X (numpy.array): normalized numerical columns
        """

        for col in range(X.shape[1]):
            X[:, col] = self._transform_col(X[:, col], col)
            
        return X

    def fit_transform(self, X, y=None):
        """Normalize numerical columns.
        
        Args:
            X (numpy.array) : numerical columns to normalize

        Returns:
            X (numpy.array): normalized numerical columns
        """

        self.ecdfs = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.ecdfs[col] = ECDF(X[:, col])
            X[:, col] = self._transform_col(X[:, col], col)

        return X

    def _transform_col(self, x, col):
        """Normalize one numerical column.
        
        Args:
            x (numpy.array): a numerical column to normalize
            col (int): column index

        Returns:
            A normalized feature vector.
        """

        return norm.ppf(self.ecdfs[col](x) * .998 + .001)


class LabelEncoder(base.BaseEstimator):
    """Label Encoder that groups infrequent values into one label.

    Attributes:
        min_obs (int): minimum number of observation to assign a label.
        label_encoders (list of dict): label encoders for columns
        label_maxes (list of int): maximum of labels for columns
    """

    def __init__(self, min_obs=10):
        """Initialize the OneHotEncoder class object.

        Args:
            min_obs (int): minimum number of observation to assign a label.
        """

        self.min_obs = min_obs

    def __repr__(self):
        return ('LabelEncoder(min_obs={})').format(self.min_obs)

    def _get_label_encoder_and_max(self, x):
        """Return a mapping from values and its maximum of a column to integer labels.

        Args:
            x (numpy.array): a categorical column to encode.

        Returns:
            label_encoder (dict): mapping from values of features to integers
            max_label (int): maximum label
        """

        # NaN cannot be used as a key for dict. So replace it with a random integer.
        x[pd.isnull(x)] = NAN_INT

        # count each unique value
        label_count = {}
        for label in x:
            try:
                label_count[label] += 1
            except KeyError:
                label_count[label] = 1

        # add unique values appearing more than min_obs to the encoder.
        label_encoder = {}
        label_index = 1
        labels_not_encoded = 0
        for label in label_count.keys():
            if label_count[label] >= self.min_obs:
                label_encoder[label] = label_index
                label_index += 1
            else:
                labels_not_encoded += 1

        max_label = label_index - 1

        # if every label is encoded, then replace the maximum label with 0 so
        # that total number of labels encoded is (# of total labels - 1).
        if labels_not_encoded == 0:
            for label in label_encoder:
                # find the label with the maximum encoded value
                if label_encoder[label] == max_label:
                    # set the value of the label to 0 and decrease the maximum
                    # by 1.
                    label_encoder[label] = 0
                    max_label -= 1
                    break

        return label_encoder, max_label

    def _transform_col(self, x, col):
        """Encode one categorical column into labels.

        Args:
            x (numpy.array): a categorical column to encode
            col (int): column index

        Returns:
            x (numpy.array): a column with labels.
        """

        label_encoder = self.label_encoders[col]

        # replace NaNs with the pre-defined random integer
        x[pd.isnull(x)] = NAN_INT

        labels = np.zeros((x.shape[0], ), dtype=np.int64)
        for label in label_encoder:
            labels[x == label] = label_encoder[label]

        return labels

    def fit(self, X, y=None):
        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.label_encoders[col], self.label_maxes[col] = \
                self._get_label_encoder_and_max(X[:, col])

        return self

    def transform(self, X):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (numpy.array): categorical columns to encode

        Returns:
            X (numpy.array): label encoded columns
        """

        for col in range(X.shape[1]):
            X[:, col] = self._transform_col(X[:, col], col)

        return X

    def fit_transform(self, X, y=None):
        """Encode categorical columns into label encoded columns

        Args:
            X (numpy.array): categorical columns to encode

        Returns:
            X (numpy.array): label encoded columns
        """

        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.label_encoders[col], self.label_maxes[col] = \
                self._get_label_encoder_and_max(X[:, col])

            X[:, col] = self._transform_col(X[:, col], col)

        return X


class OneHotEncoder(base.BaseEstimator):
    """One-Hot-Encoder that groups infrequent values into one dummy variable.

    Attributes:
        min_obs (int): minimum number of observation to create a dummy variable
        label_encoders (list of (dict, int)): label encoders and their maximums
                                              for columns
    """

    def __init__(self, min_obs=10):
        """Initialize the OneHotEncoder class object.

        Args:
            min_obs (int): minimum number of observation to create a dummy variable
            label_encoder (LabelEncoder): LabelEncoder that transofrm
        """

        self.min_obs = min_obs
        self.label_encoder = LabelEncoder(min_obs)

    def __repr__(self):
        return ('OneHotEncoder(min_obs={})').format(self.min_obs)

    def _transform_col(self, x, col):
        """Encode one categorical column into sparse matrix with one-hot-encoding.

        Args:
            x (numpy.array): a categorical column to encode
            col (int): column index

        Returns:
            X (scipy.sparse.coo_matrix): sparse matrix encoding a categorical
                                         variable into dummy variables
        """

        labels = self.label_encoder._transform_col(x, col)
        label_max = self.label_encoder.label_maxes[col]

        # build row and column index for non-zero values of a sparse matrix
        index = np.array(range(len(labels)))
        i = index[labels > 0]
        j = labels[labels > 0] - 1  # column index starts from 0

        if len(i) > 0:
            return sparse.coo_matrix((np.ones_like(i), (i, j)),
                                     shape=(x.shape[0], label_max))
        else:
            # if there is no non-zero value, return no matrix
            return None

    def fit(self, X, y=None):
        self.label_encoder.fit(X)

        return self

    def transform(self, X):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (numpy.array): categorical columns to encode

        Returns:
            X_new (scipy.sparse.coo_matrix): sparse matrix encoding categorical
                                             variables into dummy variables
        """

        for col in range(X.shape[1]):
            X_col = self._transform_col(X[:, col], col)
            if X_col is not None:
                if col == 0:
                    X_new = X_col
                else:
                    X_new = sparse.hstack((X_new, X_col))

            logging.debug('{} --> {} features'.format(
                col, self.label_encoder.label_maxes[col])
            )

        return X_new

    def fit_transform(self, X, y=None):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (numpy.array): categorical columns to encode

        Returns:
            sparse matrix encoding categorical variables into dummy variables
        """

        self.label_encoder.fit(X)

        return self.transform(X)


class BandpassFilter(base.BaseEstimator):

    def __init__(self, fs=10., lowcut=.5, highcut=3., order=3):
        self.fs = 10.
        self.lowcut = .5
        self.highcut = 3.
        self.order = 3
        b, a = _butter_bandpass()
        self.a = a
        self.b = b

    def _butter_bandpass(self):
        nyq = .5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')

        return b, a

    def _butter_bandpass_filter(self, x):
        return lfilter(self.b, self.a, x)

    def fit(self, X):
        return self

    def transform(self, X, y=None):
        for col in range(X.shape[1]):
            X[:, col] = self._butter_bandpass_filter(X[:, col])

        return X
