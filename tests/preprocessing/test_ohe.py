import cProfile
import numpy as np
import pandas as pd
from kaggler.preprocessing import OneHotEncoder


N_OBS = int(1e6)
N_FEATURE = 10
N_CATEGORY = 1000


def test():
    df = pd.DataFrame(
        np.random.randint(0, N_CATEGORY, size=(N_OBS, N_FEATURE)),
        columns=["c{}".format(x) for x in range(N_FEATURE)],
    )
    profiler = cProfile.Profile(subcalls=True, builtins=True, timeunit=0.001)
    ohe = OneHotEncoder(min_obs=100)
    profiler.enable()
    ohe.fit(df)
    X_new = ohe.transform(df)
    profiler.disable()
    profiler.print_stats()
    print("{} --> {}".format(df.shape, X_new.shape))


if __name__ == "__main__":
    test()
