"""
This code is based on the solution of the team AvengersEnsmbl at
the KDDCup 2019 AutoML track (https://github.com/jeongyoonlee/kddcup2019track2)

Details and winners' solutions at the competition are available at
the competition website (https://www.4paradigm.com/competition/kddcup2019).
"""

import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import lightgbm as lgb
from logging import getLogger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBModel

from ..const import RANDOM_SEED


logger = getLogger(__name__)
SAMPLE_SIZE = 10000
VALID_SIZE = .2


def sample_data(X, y, nrows, shuffle=True, random_state=None):
    """Sample data by rows.

    Args:
        X (pandas.DataFrame): features
        y (pandas.Series): labels
        nrows (int): the number rows to be sampled
        shuffle (bool): whether to shuffle the data before sampling or not
        random_state (None, int, or numpy.random.RandomState): random seed or a RandomState instance

    Returns:
        a tuple of:

          - X_s (pandas.DataFrame): sampled features
          - y_s (pandas.Series): sampled labels
    """
    if X.shape[0] > nrows:
        if shuffle:
            X_s = X.sample(nrows, random_state=random_state).copy()
            y_s = y[X_s.index].copy()
        else:
            X_s = X.iloc[-nrows:].copy()
            y_s = y.iloc[-nrows:].copy()
    else:
        X_s = X.copy()
        y_s = y.copy()

    return X_s, y_s


class BaseAutoML(object):
    """Base optimized regressor class."""

    def __init__(self, params, space, n_est=500, n_stop=10, sample_size=SAMPLE_SIZE, valid_size=VALID_SIZE,
                 shuffle=True, feature_selection=True, n_fs=10, fs_th=0., fs_pct=.0, hyperparam_opt=True,
                 n_hpopt=100, minimize=True, n_random_col=10, random_state=RANDOM_SEED):
        """Initialize an optimized regressor class object.

        Args:
            params (dict): default parameters for a regressor
            space (dict): parameter space for hyperopt to explore
            n_est (int): the number of iterations for a regressor
            n_stop (int): early stopping rounds for a regressor
            sample_size (int): the number of samples for feature selection and parameter search
            valid_size (float): the fraction of samples for feature selection and/or hyperparameter tuning
            shuffle (bool): if true, it uses random sampling for sampling and training/validation split. Otherwise
                last sample_size and valid_size will be used.
            feature_selection (bool): whether to select features
            n_fs (int): the number of iterations for feature selection
            fs_th (float): the feature importance threshold. Features with importances higher than it will be selected.
            fs_pct (float): the feature importance percentile. Features with importances higher than bottom x% of ranom
                features
            hyperparam_opt (bool): whether to search optimal parameters
            n_hpopt (int): the number of iterations for hyper-parameter optimization
            minimize (bool): whether the lower the metric is the better
            n_random_col (int): the number of random columns to added for feature selection
            random_state (None, int, or numpy.random.RandomState): random seed or a RandomState instance
        """

        self.params = params
        self.space = space
        for param in [p for p in params if p in self.space]:
            del self.space[param]

        self.n_est = n_est
        self.n_stop = n_stop
        self.n_fs = n_fs
        self.n_hpopt = n_hpopt
        self.sample_size = sample_size
        self.valid_size = valid_size
        self.shuffle = True
        self.feature_selection = feature_selection
        self.fs_th = fs_th
        self.fs_pct = fs_pct
        self.hyperparam_opt = hyperparam_opt
        if minimize:
            self.loss_sign = 1
        else:
            self.loss_sign = -1

        self.n_random_col = n_random_col
        if random_state is None or isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            raise ValueError('Invalid input for random_state: {}'.format(random_state))

        self.n_best = -1
        self.model = None
        self.features = []

    def tune(self, X, y):
        """Tune the regressor with feature selection and parameter search.

        Args:
            X (pandas.DataFrame): features
            y (pandas.Series): labels

        Returns:
            self
        """
        if self.feature_selection or self.hyperparam_opt:
            X_s, y_s = sample_data(X, y, self.sample_size, shuffle=self.shuffle, random_state=self.random_state)

        if self.feature_selection:
            self.features = self.select_features(X_s, y_s)
            logger.info(f'selecting top {len(self.features)} out of {X.shape[1]} features')
        else:
            self.features = X.columns.tolist()

        if self.hyperparam_opt:
            logger.info('hyper-parameter tuning')
            hyperparams, trials = self.optimize_hyperparam(X_s[self.features].values,
                                                           y_s.values,
                                                           n_eval=self.n_hpopt)

            self.params.update(hyperparams)
            self.n_best = trials.best_trial['result']['model'].best_iteration
            logger.info(f'best parameters: {self.params}')
            logger.info(f'best iterations: {self.n_best}')

        return self

    @staticmethod
    def get_feature_importance(model):
        raise NotImplementedError

    def feature_importance(self):
        raise NotImplementedError

    def select_features(self, X, y):
        """Select features based on feature importances.

        It adds self.n_random_col random columns to features and trains the regressor for
        n_eval rounds. The features ranked higher than the average rank of random columns
        in the best model are selected.

        Args:
            X (pandas.DataFrame): features
            y (pandas.Series): labels

        Returns:
            (list of str): the list of selected features
        """
        random_cols = []

        # trying for all features
        for i in range(1, self.n_random_col + 1):
            random_col = '__random_{}__'.format(i)
            X[random_col] = self.random_state.rand(X.shape[0])
            random_cols.append(random_col)

        _, trials = self.optimize_hyperparam(X.values, y.values, n_eval=self.n_fs)

        feature_importances = self.get_feature_importance(trials.best_trial['result']['model'])
        imp = pd.DataFrame({'feature_importances': feature_importances, 'feature_names': X.columns.tolist()})
        imp = imp.sort_values('feature_importances', ascending=False).drop_duplicates()

        if len(random_cols) == 0:
            imp = imp[imp['feature_importances'] > self.fs_th]
        else:
            imp_random = imp.loc[imp.feature_names.isin(random_cols), 'feature_importances'].values
            th = max(np.percentile(imp_random, self.fs_pct * 100), self.fs_th)
            logger.debug(f'feature importance (th={th:.2f}):\n{imp}')
            imp = imp[(imp.feature_importances > th) & ~(imp.feature_names.isin(random_cols))]

        return imp['feature_names'].tolist()

    def optimize_hyperparam(self, X, y, test_size=.2, n_eval=100):
        raise NotImplementedError


class AutoXGB(BaseAutoML):

    params = {'random_state': RANDOM_SEED,
              'n_jobs': -1}

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "max_depth": hp.choice("num_leaves", [6, 8, 10]),
        "colsample_bytree": hp.quniform("colsample_bytree", .5, .9, 0.1),
        "subsample": hp.quniform("subsample", .5, .9, 0.1),
        "min_child_weight": hp.choice('min_child_weight', [10, 25, 100]),
    }

    def __init__(self, objective='reg:linear', metric='rmse', boosting='gbtree', params=params, space=space,
                 n_est=500, n_stop=10, sample_size=SAMPLE_SIZE, feature_selection=True, n_fs=10, fs_th=1e-5, fs_pct=.1,
                 hyperparam_opt=True, n_hpopt=100, n_random_col=10, random_state=RANDOM_SEED, shuffle=True):

        self.metric, minimize = self._get_metric_alias_minimize(metric)

        self.params.update(params)
        self.params.update({'objective': objective,
                            'booster': boosting})

        super(AutoXGB, self).__init__(params=self.params, space=space, n_est=n_est, n_stop=n_stop,
                                      sample_size=sample_size, feature_selection=feature_selection, n_fs=n_fs,
                                      fs_th=fs_th, fs_pct=fs_pct, hyperparam_opt=hyperparam_opt, n_hpopt=n_hpopt,
                                      minimize=minimize, n_random_col=n_random_col, random_state=random_state,
                                      shuffle=shuffle)

    @staticmethod
    def _get_metric_alias_minimize(metric):
        """Get XGBoost metric alias.

        As defined at https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters

        Args:
            metric (str): a metric name

        Returns:
            (tuple):

                - (str): the standard metric name for LightGBM
                - (bool): a flag whether to minimize or maximize the metric
        """

        assert metric in ['rmse', 'rmsle', 'mae', 'logloss', 'error', 'merror', 'mlogloss', 'auc', 'aucpr',
                          'ndcg', 'map', 'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik', 'gamma-deviance',
                          'tweedie-nloglik'], 'Invalid metric: {}'.format(metric)

        if metric in ['auc', 'aucpr', 'ndcg', 'map']:
            minimize = False
        else:
            minimize = True

        return metric, minimize

    @staticmethod
    def get_feature_importance(model):
        return model.feature_importances_

    def feature_importance(self):
        return self.model.feature_importances_

    def optimize_hyperparam(self, X, y, test_size=.2, n_eval=100):
        X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=test_size, shuffle=self.shuffle)

        def objective(hyperparams):
            model = XGBModel(n_estimators=self.n_est, **self.params, **hyperparams)
            model.fit(X=X_trn, y=y_trn,
                      eval_set=[(X_val, y_val)],
                      eval_metric=self.metric,
                      early_stopping_rounds=self.n_stop,
                      verbose=False)
            score = model.evals_result()['validation_0'][self.metric][model.best_iteration] * self.loss_sign

            return {'loss': score, 'status': STATUS_OK, 'model': model}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=self.space, trials=trials,
                             algo=tpe.suggest, max_evals=n_eval, verbose=1,
                             rstate=self.random_state)

        hyperparams = space_eval(self.space, best)
        return hyperparams, trials

    def fit(self, X, y):
        self.model = XGBModel(n_estimators=self.n_best, **self.params)
        self.model.fit(X=X[self.features], y=y, eval_metric='mae', verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])


class AutoLGB(BaseAutoML):

    params = {
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "num_threads": -1,
        "feature_pre_filter": False,
    }

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127, 255]),
        "max_depth": hp.choice("max_depth", [-1, 4, 6, 8, 10]),
        "feature_fraction": hp.quniform("feature_fraction", .5, .9, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", .5, .9, 0.1),
        "min_child_samples": hp.choice('min_child_samples', [10, 25, 100]),
        "lambda_l1": hp.choice('lambda_l1', [0, .1, 1, 10]),
        "lambda_l2": hp.choice('lambda_l2', [0, .1, 1, 10]),
    }

    def __init__(self, objective='regression', metric='mae', boosting='gbdt', params=params, space=space,
                 n_est=500, n_stop=10, sample_size=SAMPLE_SIZE, feature_selection=True, n_fs=10, fs_th=1e-5, fs_pct=.1,
                 hyperparam_opt=True, n_hpopt=100, n_random_col=10, random_state=RANDOM_SEED, shuffle=True):

        self.metric, minimize = self._get_metric_alias_minimize(metric)

        self.params.update(params)
        self.params.update({'objective': objective,
                            'metric': self.metric,
                            'boosting': boosting})
        super(AutoLGB, self).__init__(params=self.params, space=space, n_est=n_est, n_stop=n_stop,
                                      sample_size=sample_size, feature_selection=feature_selection, n_fs=n_fs,
                                      fs_th=fs_th, fs_pct=fs_pct, hyperparam_opt=hyperparam_opt, n_hpopt=n_hpopt,
                                      minimize=minimize, n_random_col=n_random_col, random_state=random_state,
                                      shuffle=shuffle)

    @staticmethod
    def _get_metric_alias_minimize(metric):
        """Get LightGBM metric alias.

        As defined at https://lightgbm.readthedocs.io/en/latest/Parameters.html

        Args:
            metric (str): a metric name

        Returns:
            (tuple):

                - (str): the standard metric name for LightGBM
                - (bool): a flag whether to minimize or maximize the metric
        """

        if metric in ['l1', 'l2', 'rmse', 'quantile', 'mape', 'huber', 'fair', 'poisson', 'gamma', 'gamma_deviance',
                      'tweedie', 'ndcg', 'map', 'auc', 'binary_logloss', 'binary_error', 'multi_logloss',
                      'multi_error', 'cross_entropy', 'cross_entropy_lambda', 'kullerback_leibler']:
            pass
        elif metric in ['mae', 'mean_absolute_error', 'regression_l1']:
            metric = 'l1'
        elif metric in ['mean_squared_error', 'mse', 'regression_l2', 'regression']:
            metric = 'l2'
        elif metric in ['root_mean_squared_error', 'l2_root']:
            metric = 'rmse'
        elif metric in ['mean_absolute_percentage_error']:
            metric = 'mape'
        elif metric in ['lamdarank']:
            metric = 'ndcg'
        elif metric in ['mean_average_precision']:
            metric = 'map'
        elif metric in ['binary']:
            metric = 'binary_logloss'
        elif metric in ['multiclass', 'softmax', 'multiclassova', 'multiclass_ova', 'ova', 'ovr']:
            metric = 'multi_logloss'
        elif metric in ['xentropy']:
            metric = 'cross_entropy'
        elif metric in ['xentlambda']:
            metric = 'cross_entropy_lambda'
        elif metric in ['kldiv']:
            metric = 'kullback_leibler'
        else:
            raise ValueError('{} is not a valid metric. See https://lightgbm.readthedocs.io/en/latest/Parameters.html '
                             'for the full list of metrics available.'.format(metric))

        if metric in ['auc', 'ndcg', 'map']:
            minimize = False
        else:
            minimize = True

        return metric, minimize

    @staticmethod
    def get_feature_importance(model):
        return model.feature_importance(importance_type='gain')

    def feature_importance(self):
        return self.model.feature_importance(importance_type='gain')

    def optimize_hyperparam(self, X, y, test_size=.2, n_eval=100):
        X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=test_size, shuffle=self.shuffle)

        train_data = lgb.Dataset(X_trn, label=y_trn)
        valid_data = lgb.Dataset(X_val, label=y_val)

        def objective(hyperparams):
            model = lgb.train({**self.params, **hyperparams}, train_data, self.n_est,
                              valid_data, early_stopping_rounds=self.n_stop, verbose_eval=0)

            score = model.best_score["valid_0"][self.metric] * self.loss_sign

            return {'loss': score, 'status': STATUS_OK, 'model': model}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=self.space, trials=trials,
                             algo=tpe.suggest, max_evals=n_eval, verbose=1,
                             rstate=self.random_state)

        hyperparams = space_eval(self.space, best)
        return hyperparams, trials

    def fit(self, X, y):
        train_data = lgb.Dataset(X[self.features], label=y)
        self.model = lgb.train(self.params, train_data, self.n_best, verbose_eval=100)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features], num_iteration=self.n_best)
