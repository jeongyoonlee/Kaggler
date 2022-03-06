import logging
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from kaggler.metrics import auc, mae, rmse
from kaggler.model import AutoLGB, AutoXGB


RANDOM_SEED = 42
N_OBS = 1000
N_FEATURE = 10
N_IMP_FEATURE = 2

logging.basicConfig(level=logging.DEBUG)


def test_automl_regression():
    X, y = make_regression(
        n_samples=N_OBS,
        n_features=N_FEATURE,
        n_informative=N_IMP_FEATURE,
        random_state=RANDOM_SEED,
    )
    X = pd.DataFrame(X, columns=["x{}".format(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    logging.info(f"X dim: {X.shape}, y dim: {y.shape}")

    X_trn, X_tst, y_trn, y_tst = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = AutoLGB(objective="regression", metric="l1")
    model.tune(X_trn, y_trn)
    model.fit(X_trn, y_trn)
    p = model.predict(X_tst)
    r = np.random.rand(X_tst.shape[0]) * (y_trn.max() - y_trn.min()) + y_trn.min()
    logging.info(f"MAE (LGB): {mae(y_tst, p):.4f}")
    assert mae(y_tst, p) < mae(y_tst, r)

    model = AutoXGB(objective="reg:squarederror", metric="rmse")
    model.tune(X_trn, y_trn)
    model.fit(X_trn, y_trn)
    p = model.predict(X_tst)
    r = np.random.rand(X_tst.shape[0]) * (y_trn.max() - y_trn.min()) + y_trn.min()
    logging.info(f"MAE (RMSE): {rmse(y_tst, p):.4f}")
    assert mae(y_tst, p) < mae(y_tst, r)


def test_automl_classification():
    X, y = make_classification(
        n_samples=N_OBS,
        n_features=N_FEATURE,
        n_informative=N_IMP_FEATURE,
        random_state=RANDOM_SEED,
    )
    X = pd.DataFrame(X, columns=["x{}".format(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    logging.info(f"X dim: {X.shape}, y dim: {y.shape}")

    X_trn, X_tst, y_trn, y_tst = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = AutoLGB(objective="binary", metric="auc")
    model.tune(X_trn, y_trn)
    model.fit(X_trn, y_trn)
    p = model.predict(X_tst)
    logging.info(f"AUC (LGB): {auc(y_tst, p):.4f}")
    assert auc(y_tst, p) > 0.5

    model = AutoXGB(objective="binary:logistic", metric="auc")
    model.tune(X_trn, y_trn)
    model.fit(X_trn, y_trn)
    p = model.predict(X_tst)
    logging.info(f"AUC (XGB): {auc(y_tst, p):.4f}")
    assert auc(y_tst, p) > 0.5
