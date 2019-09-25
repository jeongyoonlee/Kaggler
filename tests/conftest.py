import numpy as np
import pandas as pd
import pytest

from .const import RANDOM_SEED, TARGET_COL


N_CATEGORY = 50
N_OBS = 10000
N_CAT_FEATURE = 10
N_NUM_FEATURE = 10


@pytest.fixture(scope='module')
def generate_data():

    generated = False

    def _generate_data():
        if not generated:
            np.random.seed(RANDOM_SEED)
            df = pd.DataFrame(
                np.hstack((
                    np.random.rand(N_OBS, N_NUM_FEATURE),
                    np.random.randint(0, N_CATEGORY, size=(N_OBS, N_CAT_FEATURE))
                )),
                columns=['num_{}'.format(x) for x in range(N_NUM_FEATURE)]
                + ['cat_{}'.format(x) for x in range(N_CAT_FEATURE)]
            )
            df[TARGET_COL] = np.random.rand(N_OBS)

        return df

    yield _generate_data
