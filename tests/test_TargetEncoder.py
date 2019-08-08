from kaggler.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


N_CATEGORY = 100
N_OBS = 10000
N_FEATURE = 10
TARGET_COL = 'target'
N_FOLD = 5
RANDOM_SEED = 42


def test_TargetEncoder():
    df = pd.DataFrame(np.random.randint(0, N_CATEGORY, size=(N_OBS, N_FEATURE)),
                      columns=['c{}'.format(x) for x in range(N_FEATURE)])
    feature_cols = df.columns
    df[TARGET_COL] = np.random.rand(N_OBS)

    te = TargetEncoder()
    X_cat = te.fit_transform(df[feature_cols], df[TARGET_COL])
    print('Without CV:\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(feature_cols)

    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_SEED)
    te = TargetEncoder(cv=cv)
    X_cat = te.fit_transform(df[feature_cols], df[TARGET_COL])
    print('With CV (fit_transform()):\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(feature_cols)

    te = TargetEncoder(cv=cv)
    te.fit(df[feature_cols], df[TARGET_COL])
    X_cat = te.transform(df[feature_cols])
    print('With CV (fit() and transform() separately):\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(feature_cols)
