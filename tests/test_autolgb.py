import logging
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from kaggler.metrics import auc, mae
from kaggler.model import AutoLGB


RANDOM_SEED = 42
N_OBS = 1000
N_FEATURE = 20
N_IMP_FEATURE = 10

logging.basicConfig(level=logging.DEBUG)


def test_autolgb_classification():
    X, y = make_classification(n_samples=N_OBS,
                               n_features=N_FEATURE,
                               n_informative=N_IMP_FEATURE,
                               random_state=RANDOM_SEED)
    X = pd.DataFrame(X, columns=['x{}'.format(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    logging.info(X.shape, y.shape)

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                                  test_size=.2,
                                                  random_state=RANDOM_SEED)

    model = AutoLGB(objective='binary', metric='auc')
    model.tune(X_trn, y_trn)
    model.fit(X_trn, y_trn)
    p = model.predict(X_tst)
    logging.info('AUC: {:.4f}'.format(auc(y_tst, p)))
    assert auc(y_tst, p) > .5


def test_autolgb_regression():
    X, y = make_regression(n_samples=N_OBS,
                           n_features=N_FEATURE,
                           n_informative=N_IMP_FEATURE,
                           random_state=RANDOM_SEED)
    X = pd.DataFrame(X, columns=['x{}'.format(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    logging.info(X.shape, y.shape)

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                                  test_size=.2,
                                                  random_state=RANDOM_SEED)

    model = AutoLGB(objective='regression', metric='l1')
    model.tune(X_trn, y_trn)
    model.fit(X_trn, y_trn)
    p = model.predict(X_tst)
    r = (np.random.rand(X_tst.shape[0]) * (y_trn.max() - y_trn.min()) +
         y_trn.min())
    logging.info('MAE: {:.4f}'.format(mae(y_tst, p)))
    assert mae(y_tst, p) < mae(y_tst, r)


def test_autolgb_get_metric_alias_minimize():
    # Test if AutoLGB can take a metric alias instead of the standard metric name
    _ = AutoLGB(objective='regression', metric='mae')

    # Test if AutoLGB raises a ValueError for unknown metrics
    with pytest.raises(ValueError):
        _ = AutoLGB(objective='regression', metric='unknown_metric')
