import logging
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from kaggler.metrics import auc
from kaggler.model import AutoLGB


RANDOM_SEED = 42
N_OBS = 10000
N_FEATURE = 100
N_IMP_FEATURE = 20


logging.basicConfig(level=logging.DEBUG)


def test_autolgb():
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
