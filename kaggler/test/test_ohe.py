from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cProfile
import numpy as np
import pandas as pd
from kaggler.preprocessing import OneHotEncoder


DATA_NUM = 1e6


def test():
    df = pd.DataFrame(np.random.randint(0, 1000, size=(1000000, 10)),
                     columns=['c{}'.format(x) for x in range(10)])
    profiler = cProfile.Profile(subcalls=True, builtins=True, timeunit=.001)
    ohe = OneHotEncoder(min_obs=100)
    profiler.enable()
    ohe.fit(df)
    X_new = ohe.transform(df)
    profiler.disable()
    profiler.print_stats()
    print('{} --> {}'.format(df.shape, X_new.shape))


if __name__=='__main__':
    test()

