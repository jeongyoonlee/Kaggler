from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cProfile
import numpy as np
import pandas as pd
from kaggler.preprocessing import LabelEncoder


N_OBS = int(1e6)
N_FEATURE = 10
N_CATEGORY = 1000


def test():
    df = pd.DataFrame(np.random.randint(0, N_CATEGORY,
                                        size=(N_OBS, N_FEATURE)),
                      columns=['c{}'.format(x) for x in range(N_FEATURE)])
    profiler = cProfile.Profile(subcalls=True, builtins=True, timeunit=.001)
    lbe = LabelEncoder(min_obs=100)
    profiler.enable()
    lbe.fit(df)
    _ = lbe.transform(df)
    profiler.disable()
    profiler.print_stats()


if __name__ == '__main__':
    test()
