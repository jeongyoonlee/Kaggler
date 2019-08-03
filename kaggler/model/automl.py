import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import lightgbm as lgb
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..const import RANDOM_SEED


logger = logging.getLogger('kaggler')
SAMPLE_SIZE = 10000


def sample_data(X, y, nrows, random_state=None):
    """Sample data by rows.

    Args:
        X (pandas.DataFrame): features
        y (pandas.Series): labels
        nrows (int): the number rows to be sampled
        random_state (int, or numpy.random.RandomState): random seed or
            a RandomState instance

    Returns:
        a tuple of:

          - X_s (pandas.DataFrame): sampled features
          - y_s (pandas.Series): sampled labels
    """
    if len(X) > nrows:
        X_s = X.sample(nrows, random_state=random_state).copy()
        y_s = y[X_s.index].copy()
    else:
        X_s = X.copy()
        y_s = y.copy()

    return X_s, y_s


class BaseAutoML(object):
    """Base Auto ML class."""

    def __init__(self, params, space, n_est=500, n_stop=10,
                 sample_size=SAMPLE_SIZE, feature_selection=True, n_fs=10,
                 hyperparam_opt=True, n_hpopt=100, minimize=True,
                 n_random_col=10, random_state=RANDOM_SEED):
        """Initialize an optimized regressor class object.

        Args:
            params (dict): default parameters for a regressor
            space (dict): parameter space for hyperopt to explore
            n_est (int): the number of iterations for a regressor
            n_stop (int): early stopping rounds for a regressor
            sample_size (int): the number of samples for feature selection and
                parameter search
            feature_selection (bool): whether to select features
            n_fs (int): the number of iterations for feature selection
            hyperparam_opt (bool): whether to search optimal parameters
            n_hpopt (int): the number of iterations for parameter search
            minimize (bool): whether the lower the metric is the better
            n_random_col (int): the number of random columns to be added for
                feature selection
            random_state (int, or numpy.random.RandomState): random seed or
                a RandomState instance
        """

        self.params = params
        self.space = space
        self.n_est = n_est
        self.n_stop = n_stop
        self.n_fs = n_fs
        self.n_hpopt = n_hpopt
        self.sample_size = sample_size
        self.feature_selection = feature_selection
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
            raise ValueError('Invalid random_state: {}'.format(random_state))

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
            X_s, y_s = sample_data(X, y, self.sample_size)

        if self.feature_selection:
            self.features = self.select_features(X_s,
                                                 y_s,
                                                 n_eval=self.n_fs)
            logger.info('selecting {} out of {} features'.format(
                len(self.features), X.shape[1])
            )
        else:
            self.features = X.columns.tolist()

        if self.hyperparam_opt:
            logger.info('hyper-parameter tuning')
            hps, trials = self.optimize_hyperparam(X_s[self.features].values,
                                                   y_s.values,
                                                   n_eval=self.n_hpopt)

            self.params.update(hps)
            self.n_best = trials.best_trial['result']['model'].best_iteration
            logger.info('best parameters: {}'.format(self.params))
            logger.info('best iterations: {}'.format(self.n_best))

        return self

    @staticmethod
    def _get_feature_importance(model):
        raise NotImplementedError

    def feature_importance(self):
        raise NotImplementedError

    def select_features(self, X, y, n_eval=10):
        """Select features based on feature importances.

        It adds self.n_random_col random columns to features and trains the
        regressor for n_eval rounds. The features ranked higher than the
        average rank of random columns in the best model are selected.

        Args:
            X (pandas.DataFrame): features
            y (pandas.Series): labels
            n_eval (int): the number of iterations for hyperopt

        Returns:
            (list of str): the list of selected features
        """
        random_cols = []

        # trying for all features
        for i in range(1, self.n_random_col + 1):
            random_col = '__random_{}__'.format(i)
            X[random_col] = self.random_state.rand(X.shape[0])
            random_cols.append(random_col)

        _, trials = self.optimize_hyperparam(X.values, y.values, n_eval=n_eval)

        feature_importances = self._get_feature_importance(
            trials.best_trial['result']['model']
        )
        imp = pd.DataFrame({'feature_importances': feature_importances,
                            'feature_names': X.columns.tolist()})
        imp = imp.sort_values('feature_importances', ascending=False)

        if len(random_cols) == 0:
            imp = imp[imp['feature_importances'] != 0]
        else:
            th = imp.loc[imp.feature_names.isin(random_cols),
                         'feature_importances'].mean()
            logger.debug('feature importance (th={:.2f}):\n{}'.format(th, imp))
            imp = imp[(imp.feature_importances > th) &
                      ~(imp.feature_names.isin(random_cols))]

        return imp['feature_names'].tolist()

    def optimize_hyperparam(self, X, y, test_size=.2, n_eval=100):
        raise NotImplementedError


class AutoLGB(BaseAutoML):

    params = {
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "num_threads": -1,
    }

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01),
                                       np.log(0.1)),
        "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127]),
        "max_depth": hp.choice("max_depth", [-1, 4, 6, 8]),
        "feature_fraction": hp.quniform("feature_fraction", .5, .8, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", .5, .8, 0.1),
        "min_child_samples": hp.choice('min_child_samples', [10, 25, 100]),
        "lambda_l1": hp.choice('lambda_l1', [.1, 1, 10]),
    }

    def __init__(self, objective='regression', metric='mae',
                 boosting='gbdt', space=space, n_est=500, n_stop=10,
                 sample_size=SAMPLE_SIZE, feature_selection=True, n_fs=10,
                 hyperparam_opt=True, n_hpopt=100, minimize=True,
                 n_random_col=10, random_state=RANDOM_SEED):

        params = AutoLGB.params
        params.update({'objective': objective,
                       'metric': metric,
                       'boosting': boosting})

        super(AutoLGB, self).__init__(
            params=params, space=space, n_est=n_est, n_stop=n_stop,
            sample_size=sample_size, feature_selection=feature_selection,
            n_fs=n_fs, hyperparam_opt=hyperparam_opt, n_hpopt=n_hpopt,
            minimize=minimize, n_random_col=n_random_col,
            random_state=random_state
        )

    @staticmethod
    def _get_feature_importance(model):
        return model.feature_importance(importance_type='gain')

    def feature_importance(self):
        return self.model.feature_importance(importance_type='gain')

    def optimize_hyperparam(self, X, y, test_size=.2, n_eval=100):
        X_trn, X_val, y_trn, y_val = train_test_split(X,
                                                      y,
                                                      test_size=test_size)

        train_data = lgb.Dataset(X_trn, label=y_trn)
        valid_data = lgb.Dataset(X_val, label=y_val)

        def objective(hyperparams):
            model = lgb.train({**self.params, **hyperparams},
                              train_data,
                              self.n_est,
                              valid_data,
                              early_stopping_rounds=self.n_stop,
                              verbose_eval=0)

            score = (model.best_score["valid_0"][self.params["metric"]] *
                     self.loss_sign)

            return {'loss': score, 'status': STATUS_OK, 'model': model}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=self.space, trials=trials,
                             algo=tpe.suggest, max_evals=n_eval, verbose=1,
                             rstate=self.random_state)

        hyperparams = space_eval(self.space, best)
        return hyperparams, trials

    def fit(self, X, y):
        train_data = lgb.Dataset(X[self.features], label=y)
        self.model = lgb.train(self.params, train_data, self.n_best,
                               verbose_eval=100)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features], num_iteration=self.n_best)
