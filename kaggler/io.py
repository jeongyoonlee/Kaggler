from sklearn.datasets import load_svmlight_file

import heapq
import numpy as np


def is_number(s):
    """Check if a string is a number or not."""

    try:
        float(s)
        return True
    except ValueError:
        return False


def load_data(path, dense=False):
    """Load data from a CSV or libsvm format file.
    
    Args:
        path: A path to the CSV or libsvm format file containing data.
        dense: An optional variable indicating if the return matrix should be
            dense.  By default, it is false.
    """

    with open(path, 'r') as f:
        line = f.readline().strip()

    if ':' in line:
        X, y = load_svmlight_file(path)
        X = X.astype(np.float32)
        if dense:
            X = X.todense()
    elif ',' in line:
        X = np.loadtxt(path, delimiter=',',
                       skiprows=0 if is_number(line.split(',')[0]) else 1)
        y = X[:, 0]
        X = X[:, 1:]
    else:
        raise NotImplementedError, "Neither CSV nor LibSVM formatted file."

    return X, y


def read_sps(path):
    for line in open(path):
        # parse x
        xs = line.rstrip().split(' ')

        yield xs[1:], int(xs[0])


def shuf_file(f, shuf_win):
    heap = []
    for line in f:
        key = hash(line)
        if len(heap) < shuf_win:
            heapq.heappush(heap, (key, line))
        else:
            _, out = heapq.heappushpop(heap, (key, line))
            yield out

    while len(heap) > 0:
        _, out = heapq.heappop(heap)
        yield out
