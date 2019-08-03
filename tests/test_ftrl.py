from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cProfile
import numpy as np
from scipy import sparse
from kaggler.online_model import FTRL
from kaggler.metrics import auc


np.random.seed(1234)
N_VALUE = int(4e6)
N_OBS = int(1e6)
N_FEATURE = 100


def main():
    print('create y...')
    y = np.random.randint(2, size=N_OBS)
    print('create X...')
    row = np.random.randint(N_OBS, size=N_VALUE)
    col = np.random.randint(N_FEATURE, size=N_VALUE)
    data = np.ones(N_VALUE)
    X = sparse.csr_matrix((data, (row, col)), dtype=np.int8)

    print('train...')
    profiler = cProfile.Profile(subcalls=True, builtins=True, timeunit=0.001,)
    clf = FTRL(interaction=False)
    profiler.enable()
    clf.fit(X, y)
    profiler.disable()
    profiler.print_stats()

    p = clf.predict(X)
    print('AUC: {:.4f}'.format(auc(y, p)))

    assert auc(y, p) > .5


if __name__ == '__main__':
    main()
