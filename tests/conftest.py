import numpy as np
import pandas as pd
import pytest

from .const import RANDOM_SEED, TARGET_COL


N_CATEGORY = 50
N_OBS = 10000
N_CAT_FEATURE = 10
N_NUM_FEATURE = 5


@pytest.fixture(scope='module')
def generate_data():

    generated = False

    def _generate_data():

        if not generated:
            assert N_CAT_FEATURE > 1
            assert N_NUM_FEATURE > 3
            np.random.seed(RANDOM_SEED)

            X_num = np.random.normal(size=(N_OBS, N_NUM_FEATURE))
            X_cat = np.random.randint(0, N_CATEGORY, size=(N_OBS, N_CAT_FEATURE))
            df = pd.DataFrame(
                np.hstack((X_num, X_cat)),
                columns=['num_{}'.format(x) for x in range(N_NUM_FEATURE)]
                + ['cat_{}'.format(x) for x in range(N_CAT_FEATURE)]
            )
            df[TARGET_COL] = (1 + X_num[:, 0] * X_num[:, 1] - np.log1p(np.exp(X_num[:, 1] + X_num[:, 2]))
                              + 10 * (X_cat[:, 0] == 0).astype(int)
                              + np.random.normal(scale=.01, size=N_OBS))

        return df

    yield _generate_data
