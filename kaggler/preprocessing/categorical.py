from logging import getLogger
import numpy as np
from scipy import sparse
from sklearn import base
from sklearn.model_selection import KFold
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Reshape, Concatenate, BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .const import EMBEDDING_SUFFIX, MIN_EMBEDDING, NAN_INT


logger = getLogger(__name__)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


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
            x (pandas.Series): a categorical column to encode.

        Returns:
            (tuple):
                - (dict): mapping from values of features to integers
                - (int): maximum label
        """

        # NaN cannot be used as a key for dict. Impute it with a random
        # integer.
        label_count = x.fillna(NAN_INT).value_counts()
        n_uniq = label_count.shape[0]

        label_count = label_count[label_count >= self.min_obs]
        n_uniq_new = label_count.shape[0]

        # If every label appears more than min_obs, new label starts from 0.
        # Otherwise, new label starts from 1 and 0 is used for all old labels
        # that appear less than min_obs.
        offset = 0 if n_uniq == n_uniq_new else 1

        label_encoder = pd.Series(np.arange(n_uniq_new) + offset,
                                  index=label_count.index)
        max_label = label_encoder.max()
        label_encoder = label_encoder.to_dict()

        return label_encoder, max_label

    def _transform_col(self, x, i):
        """Encode one categorical column into labels.

        Args:
            x (pandas.Series): a categorical column to encode
            i (int): column index

        Returns:
            (pandas.Series): a column with labels.
        """
        return x.fillna(NAN_INT).map(self.label_encoders[i]).fillna(0).astype(int)

    def fit(self, X, y=None):
        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            self.label_encoders[i], self.label_maxes[i] = \
                self._get_label_encoder_and_max(X[col])

        return self

    def transform(self, X):
        """Encode categorical columns into label encoded columns

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): label encoded columns
        """

        X = X.copy()
        for i, col in enumerate(X.columns):
            X.loc[:, col] = self._transform_col(X[col], i)

        return X

    def fit_transform(self, X, y=None):
        """Encode categorical columns into label encoded columns

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): label encoded columns
        """

        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        X = X.copy()
        for i, col in enumerate(X.columns):
            self.label_encoders[i], self.label_maxes[i] = \
                self._get_label_encoder_and_max(X[col])

            X.loc[:, col] = (X[col].fillna(NAN_INT)
                             .map(self.label_encoders[i])
                             .fillna(0).astype(int))

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
            min_obs (int): minimum number of observations required to create
                a dummy variable
            label_encoder (LabelEncoder): LabelEncoder that transofrm
        """

        self.min_obs = min_obs
        self.label_encoder = LabelEncoder(min_obs)

    def __repr__(self):
        return ('OneHotEncoder(min_obs={})').format(self.min_obs)

    def _transform_col(self, x, i):
        """Encode one categorical column into sparse matrix with one-hot-encoding.

        Args:
            x (pandas.Series): a categorical column to encode
            i (int): column index

        Returns:
            (scipy.sparse.coo_matrix): sparse matrix encoding a categorical
                                       variable into dummy variables
        """

        labels = self.label_encoder._transform_col(x, i)
        label_max = self.label_encoder.label_maxes[i]

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
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (scipy.sparse.coo_matrix): sparse matrix encoding categorical
                                       variables into dummy variables
        """

        for i, col in enumerate(X.columns):
            X_col = self._transform_col(X[col], i)
            if X_col is not None:
                if i == 0:
                    X_new = X_col
                else:
                    X_new = sparse.hstack((X_new, X_col))

            logger.debug('{} --> {} features'.format(
                col, self.label_encoder.label_maxes[i])
            )

        return X_new

    def fit_transform(self, X, y=None):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            sparse matrix encoding categorical variables into dummy variables
        """

        self.label_encoder.fit(X)

        return self.transform(X)


class TargetEncoder(base.BaseEstimator):
    """Target Encoder that encode categorical values into average target values.

    Smoothing and min_samples are added based on olivier's kernel at Kaggle:
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

    , which is based on Daniele Micci-Barreca (2001):
    https://dl.acm.org/citation.cfm?id=507538

    Attributes:
        target_encoders (list of dict): target encoders for columns
    """

    def __init__(self, smoothing=1, min_samples=10, cv=kfold):
        """Initialize the TargetEncoder class object.

        Args:
            smoothing (int): smoothing effect to balance between the categorical average vs global mean
            min_samples (int): minimum samples to take category average into account
            cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object. default=KFold(5, True, 42)
        """
        assert (min_samples >= 0) and (smoothing >= 0), 'min_samples and smoothing should be positive'
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.cv = cv

    def __repr__(self):
        return('TargetEncoder(smoothing={}, min_samples={}, cv={})'.format(self.smoothing, self.min_samples, self.cv))

    def _get_target_encoder(self, x, y):
        """Return a mapping from categories to average target values.

        Args:
            x (pandas.Series): a categorical column to encode.
            y (pandas.Series): the target column

        Returns:
            (dict): mapping from categories to average target values
        """

        assert len(x) == len(y)

        # NaN cannot be used as a key for dict. So replace it with a random
        # integer
        mean_count = pd.DataFrame({y.name: y, x.name: x.fillna(NAN_INT)}).groupby(x.name)[y.name].agg(['mean', 'count'])
        smoothing = 1 / (1 + np.exp(-(mean_count['count'] - self.min_samples) / self.smoothing))

        mean_count[y.name] = self.target_mean * (1 - smoothing) + mean_count['mean'] * smoothing
        return mean_count[y.name].to_dict()

    def fit(self, X, y):
        """Encode categorical columns into average target values.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series): the target column

        Returns:
            (pandas.DataFrame): encoded columns
        """
        self.target_encoders = [None] * X.shape[1]
        self.target_mean = y.mean()

        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.target_encoders[i] = self._get_target_encoder(X[col], y)
            else:
                self.target_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col], y), 1):
                    i_trn, i_val = X.index[i_trn], X.index[i_val]
                    self.target_encoders[i].append(self._get_target_encoder(X.loc[i_trn, col], y.loc[i_trn]))

        return self

    def transform(self, X):
        """Encode categorical columns into average target values.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): encoded columns
        """
        X = X.copy()
        for i, col in enumerate(X.columns):
            if self.cv is None:
                X.loc[:, col] = (X[col].fillna(NAN_INT)
                                       .map(self.target_encoders[i])
                                       .fillna(self.target_mean))
            else:
                for i_enc, target_encoder in enumerate(self.target_encoders[i], 1):
                    if i_enc == 1:
                        x = X[col].fillna(NAN_INT).map(target_encoder).fillna(self.target_mean)
                    else:
                        x += X[col].fillna(NAN_INT).map(target_encoder).fillna(self.target_mean)

                X.loc[:, col] = x / i_enc

        return X.astype(float)

    def fit_transform(self, X, y):
        """Encode categorical columns into average target values.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series): the target column

        Returns:
            (pandas.DataFrame): encoded columns
        """
        self.target_encoders = [None] * X.shape[1]
        self.target_mean = y.mean()

        X = X.copy()
        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.target_encoders[i] = self._get_target_encoder(X[col], y)

                X.loc[:, col] = (X[col].fillna(NAN_INT)
                                       .map(self.target_encoders[i])
                                       .fillna(self.target_mean))
            else:
                self.target_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col], y), 1):
                    i_trn, i_val = X.index[i_trn], X.index[i_val]
                    target_encoder = self._get_target_encoder(X.loc[i_trn, col], y[i_trn])

                    X.loc[i_val, col] = (X.loc[i_val, col].fillna(NAN_INT)
                                                          .map(target_encoder)
                                                          .fillna(y[i_trn].mean()))

                    self.target_encoders[i].append(target_encoder)

        return X.astype(float)


class EmbeddingEncoder(base.BaseEstimator):
    """EmbeddingEncoder encodes categorical features to numerical embedding features.

    Reference: 'Entity embeddings to handle categories' by Abhishek Thakur
    at https://www.kaggle.com/abhishek/entity-embeddings-to-handle-categories
    """

    def __init__(self, cat_cols, num_cols=[], n_emb=[], min_obs=10, n_epoch=10, batch_size=1024, cv=None,
                 random_state=42):
        """Initialize an EmbeddingEncoder class object.

        Args:
            cat_cols (list of str): the names of categorical features to create embeddings for.
            num_cols (list of str): the names of numerical features to train embeddings with.
            n_emb (int or list of int): the numbers of embedding features used for columns.
            min_obs (int): categories observed less than it will be grouped together before training embeddings
            n_epoch (int): the number of epochs to train a neural network with embedding layer
            batch_size (int): the size of mini-batches in model training
            cv (sklearn.model_selection._BaseKFold): sklearn CV object
            random_state (int): random seed.
        """
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        if isinstance(n_emb, int):
            self.n_emb = [n_emb] * len(cat_cols)
        elif isinstance(n_emb, list):
            if not n_emb:
                self.n_emb = [None] * len(cat_cols)
            else:
                assert len(cat_cols) == len(n_emb)
                self.n_emb = n_emb
        else:
            raise ValueError('n_emb should be int or list')

        self.min_obs = min_obs
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.cv = cv
        self.random_state = random_state

        self.lbe = LabelEncoder(min_obs=self.min_obs)

    @staticmethod
    def _get_model(X, cat_cols, num_cols, n_uniq, n_emb, output_activation):
        inputs = []
        num_inputs = []
        embeddings = []
        for i, col in enumerate(cat_cols):

            if not n_uniq[i]:
                n_uniq[i] = X[col].nunique()
            if not n_emb[i]:
                n_emb[i] = max(MIN_EMBEDDING, 2 * int(np.log2(n_uniq[i])))

            _input = Input(shape=(1,), name=col)
            _embed = Embedding(input_dim=n_uniq[i], output_dim=n_emb[i], name=col + EMBEDDING_SUFFIX)(_input)
            _embed = Dropout(.2)(_embed)
            _embed = Reshape((n_emb[i],))(_embed)

            inputs.append(_input)
            embeddings.append(_embed)

        if num_cols:
            num_inputs = Input(shape=(len(num_cols),), name='num_inputs')
            merged_input = Concatenate(axis=1)(embeddings + [num_inputs])

            inputs = inputs + [num_inputs]
        else:
            merged_input = Concatenate(axis=1)(embeddings)

        x = BatchNormalization()(merged_input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(.5)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(.5)(x)
        x = BatchNormalization()(x)
        output = Dense(1, activation=output_activation)(x)

        model = Model(inputs=inputs, outputs=output)

        return model, n_emb, n_uniq

    def fit(self, X, y):
        """Train a neural network model with embedding layers.

        Args:
            X (pandas.DataFrame): categorical features to create embeddings for
            y (pandas.Series): a target variable

        Returns:
            A trained EmbeddingEncoder object.
        """
        is_classification = y.nunique() == 2

        X_cat = self.lbe.fit_transform(X[self.cat_cols])
        if is_classification:
            assert np.isin(y, [0, 1]).all(), 'Target values should be 0 or 1 for classification.'
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = [AUC()]
            monitor = 'val_auc'
            mode = 'max'
        else:
            output_activation = 'linear'
            loss = 'mse'
            metrics = ['mse']
            monitor = 'val_mse'
            mode = 'min'

        n_uniq = [X_cat[col].nunique() for col in self.cat_cols]
        if self.cv:
            self.embs = []
            n_fold = self.cv.get_n_splits(X)
            for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X, y), 1):
                i_trn, i_val = X.index[i_trn], X.index[i_val]
                model, self.n_emb, _ = self._get_model(X_cat, self.cat_cols, self.num_cols, n_uniq, self.n_emb,
                                                       output_activation)
                model.compile(optimizer=Adam(lr=0.01), loss=loss, metrics=metrics)

                features_trn = [X_cat.loc[i_trn, col] for col in self.cat_cols]
                features_val = [X_cat.loc[i_val, col] for col in self.cat_cols]
                if self.num_cols:
                    features_trn += [X.loc[i_trn, self.num_cols]]
                    features_val += [X.loc[i_val, self.num_cols]]

                es = EarlyStopping(monitor=monitor, min_delta=.001, patience=5, verbose=1, mode=mode,
                                   baseline=None, restore_best_weights=True)
                rlr = ReduceLROnPlateau(monitor=monitor, factor=.5, patience=3, min_lr=1e-6, mode=mode)
                model.fit(x=features_trn,
                          y=y[i_trn],
                          validation_data=(features_val, y[i_val]),
                          epochs=self.n_epoch,
                          batch_size=self.batch_size,
                          callbacks=[es, rlr])

                for i_col, col in enumerate(self.cat_cols):
                    emb = model.get_layer(col + EMBEDDING_SUFFIX).get_weights()[0]
                    if i_cv == 1:
                        self.embs.append(emb / n_fold)
                    else:
                        self.embs[i_col] += emb / n_fold

        else:
            model, self.n_emb, _ = self._get_model(X_cat, self.cat_cols, self.num_cols, n_uniq, self.n_emb,
                                                   output_activation)
            model.compile(optimizer=Adam(lr=0.01), loss=loss, metrics=metrics)

            features = [X_cat[col] for col in self.cat_cols]
            if self.num_cols:
                features += [X[self.num_cols].values]

            es = EarlyStopping(monitor=monitor, min_delta=.001, patience=5, verbose=1, mode=mode,
                               baseline=None, restore_best_weights=True)
            rlr = ReduceLROnPlateau(monitor=monitor, factor=.5, patience=3, min_lr=1e-6, mode=mode)
            model.fit(x=features,
                      y=y,
                      epochs=self.n_epoch,
                      validation_split=.2,
                      batch_size=self.batch_size,
                      callbacks=[es, rlr])

            self.embs = []
            for i, col in enumerate(self.cat_cols):
                self.embs.append(model.get_layer(col + EMBEDDING_SUFFIX).get_weights()[0])
                logger.debug('{}: {}'.format(col, self.embs[i].shape))

    def transform(self, X):
        X_cat = self.lbe.transform(X[self.cat_cols])
        X_emb = []

        for i, col in enumerate(self.cat_cols):
            X_emb.append(self.embs[i][X_cat[col].values])

        return np.hstack(X_emb)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class FrequencyEncoder(base.BaseEstimator):
    """Frequency Encoder that encode categorical values by counting frequencies.

    Attributes:
        frequency_encoders (list of dict): frequency encoders for columns
    """

    def __init__(self, cv=None):
        """Initialize the FrequencyEncoder class object.
        Args:
            cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        """
        self.cv = cv

    def __repr__(self):
        return('FrequencyEncoder(cv={})'.format(self.cv))

    def _get_frequency_encoder(self, x):
        """Return a mapping from categories to frequency.

        Args:
            x (pandas.Series): a categorical column to encode.

        Returns:
            (dict): mapping from categories to frequency
        """

        # NaN cannot be used as a key for dict. So replace it with a random
        # integer
        df = pd.DataFrame({x.name: x.fillna('NaN')})
        df[x.name + '_freq'] = df[x.name].map(df[x.name].value_counts())
        return df.groupby(x.name)[x.name + '_freq'].size().to_dict()

    def fit(self, X, y=None):
        """Encode categorical columns into frequency.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series, optional): the target column

        Returns:
            (pandas.DataFrame): encoded columns
        """
        self.frequency_encoders = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.frequency_encoders[i] = self._get_frequency_encoder(X[col])
            else:
                self.frequency_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col]), 1):
                    self.frequency_encoders[i].append(self._get_frequency_encoder(X.iloc[i_trn][col]))

        return self

    def transform(self, X):
        """Encode categorical columns into feature frequency counts.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): encoded columns
        """
        X = X.copy()
        for i, col in enumerate(X.columns):
            if self.cv is None:
                X.loc[:, col] = X[col].fillna('NaN').map(self.frequency_encoders[i]).fillna(0)
            else:
                for i_enc, frequency_encoder in enumerate(self.frequency_encoders[i], 1):
                    if i_enc == 1:
                        x = X[col].fillna('NaN').map(frequency_encoder).fillna(0)
                    else:
                        x += X[col].fillna('NaN').map(frequency_encoder).fillna(0)

                X.loc[:, col] = x / i_enc

        return X

    def fit_transform(self, X, y=None):
        """Encode categorical columns into feature frequency counts.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series, optional): the target column
        """
        self.frequency_encoders = [None] * X.shape[1]

        X = X.copy()
        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.frequency_encoders[i] = self._get_frequency_encoder(X[col])

                X.loc[:, col] = X[col].fillna('NaN').map(self.frequency_encoders[i]).fillna(0)
            else:
                self.frequency_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col]), 1):
                    i_trn, i_val = X.index[i_trn], X.index[i_val]
                    frequency_encoder = self._get_frequency_encoder(X.loc[i_trn, col])

                    X.loc[i_val, col] = X.loc[i_val, col].fillna('NaN').map(frequency_encoder).fillna(0)
                    self.frequency_encoders[i].append(frequency_encoder)

        return X
