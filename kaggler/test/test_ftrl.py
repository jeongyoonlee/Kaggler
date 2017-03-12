import pickle as pkl
import cProfile

import numpy as np
np.random.seed(1234)
import scipy.sparse as sps

from kaggler.online_model import FTRL


DATA_NUM = int(4e6)


def main():
    print('create y...')
    y = np.random.randint(2, size=DATA_NUM)
    print('create x...')
    row = np.random.randint(1000000, size=DATA_NUM)
    col = np.random.randint(100, size=DATA_NUM)
    data = np.ones(DATA_NUM)
    x = sps.csr_matrix((data, (row, col)), dtype=np.int8)
    
    print('train...')
    profiler = cProfile.Profile(subcalls=True, builtins=True, timeunit=0.001,)
    clf = FTRL(interaction=False)
    profiler.enable()
    clf.fit(x, y)
    profiler.disable()
    profiler.print_stats()
    clf.predict(x)


if __name__ == '__main__':
    main()
