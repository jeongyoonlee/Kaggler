from scipy import sparse
import logging
import numpy as np
import pandas as pd


NAN_STR = '__KAGGLER_NAN_STR__'


class OneHotEncoder(object):
    """One-Hot-Encoder that groups infrequent values into one dummy variable.

    Attributes:
        min_obs (int): minimum number of observation to create a dummy variable
        nan_as_var (bool): whether to create a dummy variable for NaN or not
    """

    def __init__(self, min_obs=10, nan_as_var=False):
        """Initialize the OneHotEncoder class object.

        Args:
            min_obs (int): minimum number of observation to create a dummy variable
            nan_as_var (bool): whether to create a dummy variable for NaN or not
        """

        self.min_obs = min_obs
        self.nan_as_var = nan_as_var

    def _get_label_encoder(self, x):
        """Return a mapping from values of features to integer labels.

        Args:
            x (numpy.array): a categorical column to encode

        Returns:
            label_encoder (dict): mapping from values of features to integer labels
        """
        label_count = {}
        for label in x:
            # NaN cannot be used as a key for dict. So replace it with str.
            if pd.isnull(label):
                label = NAN_STR

            try:
                label_count[label] += 1
            except KeyError:
                label_count[label] = 1

        label_encoder = {}
        label_index = 1
        for label in label_count.keys():
            if (not self.nan_as_var) and label == NAN_STR:
                label_encoder[label] = -1
            elif label_count[label] >= self.min_obs:
                label_encoder[label] = label_index
                label_index += 1

        return label_encoder

    def _transform_col(self, x, col):
        """Encode one categorical column into sparse matrix with one-hot-encoding.

        Args:
            x (numpy.array): a categorical column to encode

        Returns:
            X (scipy.sparse.coo_matrix): sparse matrix encoding a categorical
                                             variable into dummy variables
        """


        labels = np.zeros((x.shape[0], ))
        for label in self.label_encoders[col]:
            labels[x == label] = self.label_encoders[col][label]

        index = np.array(range(len(labels)))
        if len(labels[labels == 0]) >= self.min_obs:
            i = index[labels >= 0]
            j = labels[labels >= 0]
        else:
            i = index[labels > 0]
            j = labels[labels > 0] - 1

        if len(i) > 0:
            return sparse.coo_matrix((np.ones_like(i), (i, j)),
                                     shape=(x.shape[0], j.max() + 1))
        else:
            return None

    def fit(self, X):
        self.label_encoders = [None] * X.shape[1]

        for i_col in range(X.shape[1]):
            self.label_encoders[i_col] = self._get_label_encoder(X[:, i_col])

        return self

    def transform(self, X):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (numpy.array): categorical columns to encode

        Returns:
            X_new (scipy.sparse.coo_matrix): sparse matrix encoding categorical
                                             variables into dummy variables
        """

        n_feature = 0
        for i_col in range(X.shape[1]):
            X_col = self._transform_col(X[:, i_col], i_col)
            if X_col is not None:
                if i_col == 0:
                    X_new = X_col
                else:
                    X_new = sparse.hstack((X_new, X_col))

            logging.debug('{} --> {} features'.format(i_col, X_new.shape[1] - n_feature))
            n_feature = X_new.shape[1]

        return X_new

    def fit_transform(self, X):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (numpy.array): categorical columns to encode

        Returns:
            X_new (scipy.sparse.coo_matrix): sparse matrix encoding categorical
                                             variables into dummy variables
        """

        self.label_encoders = [None] * X.shape[1]

        n_feature = 0
        for i_col in range(X.shape[1]):
            self.label_encoders[i_col] = self._get_label_encoder(X[:, i_col])

            X_col = self._transform_col(X[:, i_col], i_col)
            if X_col is not None:
                if i_col == 0:
                    X_new = X_col
                else:
                    X_new = sparse.hstack((X_new, X_col))

            logging.debug('{} --> {} features'.format(i_col, X_new.shape[1] - n_feature))
            n_feature = X_new.shape[1]

        return X_new
