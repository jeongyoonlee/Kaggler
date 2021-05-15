from logging import getLogger
import numpy as np
from scipy.signal import butter, lfilter
from scipy.stats import norm
from sklearn import base
from statsmodels.distributions.empirical_distribution import ECDF


logger = getLogger(__name__)


class QuantileEncoder(base.BaseEstimator):
    """QuantileEncoder encodes numerical features to quantile values.

    Attributes:
        ecdfs (list of empirical CDF): empirical CDFs for columns
        n_label (int): the number of labels to be created.
    """

    def __init__(self, n_label=10, sample=100000, random_state=42):
        """Initialize a QuantileEncoder class object.

        Args:
            n_label (int): the number of labels to be created.
            sample (int or float): the number or fraction of samples for ECDF
        """
        self.n_label = n_label
        self.sample = sample
        self.random_state = random_state

    def fit(self, X, y=None):
        """Get empirical CDFs of numerical features.

        Args:
            X (pandas.DataFrame): numerical features to encode

        Returns:
            A trained QuantileEncoder object.
        """
        def _calculate_ecdf(x):
            return ECDF(x[~np.isnan(x)])

        if self.sample >= X.shape[0]:
            self.ecdfs = X.apply(_calculate_ecdf, axis=0)
        elif self.sample > 1:
            self.ecdfs = X.sample(n=self.sample,
                                  random_state=self.random_state).apply(
                                      _calculate_ecdf, axis=0
                                  )
        else:
            self.ecdfs = X.sample(frac=self.sample,
                                  random_state=self.random_state).apply(
                                      _calculate_ecdf, axis=0
                                  )

        return self

    def fit_transform(self, X, y=None):
        """Get empirical CDFs of numerical features and encode to quantiles.

        Args:
            X (pandas.DataFrame): numerical features to encode

        Returns:
            Encoded features (pandas.DataFrame).
        """
        self.fit(X, y)

        return self.transform(X)

    def transform(self, X):
        """Encode numerical features to quantiles.

        Args:
            X (pandas.DataFrame): numerical features to encode

        Returns:
            Encoded features (pandas.DataFrame).
        """
        X = X.copy()
        for i, col in enumerate(X.columns):
            X.loc[:, col] = self._transform_col(X[col], i)

        return X

    def _transform_col(self, x, i):
        """Encode one numerical feature column to quantiles.

        Args:
            x (pandas.Series): numerical feature column to encode
            i (int): column index of the numerical feature

        Returns:
            Encoded feature (pandas.Series).
        """
        # Map values to the emperical CDF between .1% and 99.9%
        rv = np.ones_like(x) * -1

        filt = ~np.isnan(x)
        rv[filt] = np.floor((self.ecdfs[i](x[filt]) * 0.998 + .001) *
                            self.n_label)

        return rv


class Normalizer(base.BaseEstimator):
    """Normalizer that transforms numerical columns into normal distribution.

    Attributes:
        ecdfs (list of empirical CDF): empirical CDFs for columns
    """

    def fit(self, X, y=None):
        self.ecdfs = [None] * X.shape[1]

        for col in range(X.shape[1]):
            self.ecdfs[col] = ECDF(X[col].values)

        return self

    def transform(self, X):
        """Normalize numerical columns.

        Args:
            X (pandas.DataFrame) : numerical columns to normalize

        Returns:
            (pandas.DataFrame): normalized numerical columns
        """

        X = X.copy()
        for col in range(X.shape[1]):
            X[col] = self._transform_col(X[col], col)

        return X

    def fit_transform(self, X, y=None):
        """Normalize numerical columns.

        Args:
            X (pandas.DataFrame) : numerical columns to normalize

        Returns:
            (pandas.DataFrame): normalized numerical columns
        """

        self.ecdfs = [None] * X.shape[1]

        X = X.copy()
        for col in range(X.shape[1]):
            self.ecdfs[col] = ECDF(X[col].values)
            X[col] = self._transform_col(X[col], col)

        return X

    def _transform_col(self, x, col):
        """Normalize one numerical column.

        Args:
            x (pandas.Series): a numerical column to normalize
            col (int): column index

        Returns:
            A normalized feature vector.
        """

        return norm.ppf(self.ecdfs[col](x.values) * .998 + .001)


class BandpassFilter(base.BaseEstimator):

    def __init__(self, fs=10., lowcut=.5, highcut=3., order=3):
        self.fs = 10.
        self.lowcut = .5
        self.highcut = 3.
        self.order = 3
        self.b, self.a = self._butter_bandpass()

    def _butter_bandpass(self):
        nyq = .5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
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
