from io import open
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from scipy import sparse

import csv
import datetime
import heapq
import json
import os
import pickle
import time
import logging

import h5py
import numpy as np


logger = logging.getLogger(__name__)


def is_number(s):
    """Check if a string is a number or not."""

    try:
        float(s)
        return True
    except ValueError:
        return False


def save_data(X, y, path):
    """Save data as a CSV, LibSVM or HDF5 file based on the file extension.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector. If None, all zero vector will be saved.
        path (str): Path to the CSV, LibSVM or HDF5 file to save data.
    """
    catalog = {'.csv': save_csv, '.sps': save_libsvm, '.h5': save_hdf5}

    ext = os.path.splitext(path)[1]
    func = catalog[ext]

    if y is None:
        y = np.zeros((X.shape[0], ))

    func(X, y, path)


def save_csv(X, y, path):
    """Save data as a CSV file.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector.
        path (str): Path to the CSV file to save data.
    """

    if sparse.issparse(X):
        X = X.todense()

    np.savetxt(path, np.hstack((y.reshape((-1, 1)), X)))


def save_libsvm(X, y, path):
    """Save data as a LibSVM file.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector.
        path (str): Path to the CSV file to save data.
    """

    dump_svmlight_file(X, y, path, zero_based=False)


def save_hdf5(X, y, path):
    """Save data as a HDF5 file.

    Args:
        X (numpy or scipy sparse matrix): Data matrix
        y (numpy array): Target vector.
        path (str): Path to the HDF5 file to save data.
    """

    with h5py.File(path, 'w') as f:
        is_sparse = 1 if sparse.issparse(X) else 0
        f['issparse'] = is_sparse
        f['target'] = y

        if is_sparse:
            f['shape'] = np.array(X.shape)
            f['data'] = X.data
            f['indices'] = X.indices
            f['indptr'] = X.indptr
        else:
            f['data'] = X
 

def load_data(path, dense=False):
    """Load data from a CSV, LibSVM or HDF5 file based on the file extension.
    
    Args:
        path (str): A path to the CSV, LibSVM or HDF5 format file containing data.
        dense (boolean): An optional variable indicating if the return matrix
                         should be dense.  By default, it is false.

    Returns:
        Data matrix X and target vector y
    """

    catalog = {'.csv': load_csv, '.sps': load_svmlight_file, '.h5': load_hdf5}

    ext = os.path.splitext(path)[1]
    func = catalog[ext]
    X, y = func(path)

    if dense and sparse.issparse(X):
        X = X.todense()

    return X, y


def load_csv(path):
    """Load data from a CSV file.

    Args:
        path (str): A path to the CSV format file containing data.
        dense (boolean): An optional variable indicating if the return matrix
                         should be dense.  By default, it is false.

    Returns:
        Data matrix X and target vector y
    """

    with open(path) as f:
        line = f.readline().strip()

    X = np.loadtxt(path, delimiter=',',
                   skiprows=0 if is_number(line.split(',')[0]) else 1)

    y = np.array(X[:, 0]).flatten()
    X = X[:, 1:]

    return X, y


def load_hdf5(path):
    """Load data from a HDF5 file.

    Args:
        path (str): A path to the HDF5 format file containing data.
        dense (boolean): An optional variable indicating if the return matrix
                         should be dense.  By default, it is false.

    Returns:
        Data matrix X and target vector y
    """

    with h5py.File(path, 'r') as f:
        is_sparse = f['issparse'][...]
        if is_sparse:
            shape = tuple(f['shape'][...])
            data = f['data'][...]
            indices = f['indices'][...]
            indptr = f['indptr'][...]
            X = sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            X = f['data'][...]

        y = f['target'][...]

    return X, y


def read_sps(path):
    """Read a LibSVM file line-by-line.

    Args:
        path (str): A path to the LibSVM file to read.

    Yields:
        data (list) and target (int).
    """

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
        

"""
Code below was originally written by Baris Umog (https://www.kaggle.com/barisumog).
"""
class PathJoiner:
    """Load directory names from SETTINGS.json.

    Usage:
        # In SETTINGS.json, "data": "/path/to/data/".
        # To load "/path/to/data/targets.array" file to y:
        PATH = PathJoiner()
        y = load(PATH.data('targets.array'))
    """

    def __init__(self, filename='SETTINGS.json'):
        with open(filename) as file:
            self.subdirs = json.load(file)

    def __getattr__(self, attr):
        subdir = self.subdirs[attr]
        return lambda *dirs: os.path.join(subdir, *dirs)


def stream_lines(filename, encoding='utf-8', ignore_errors=False):
    errors = 'ignore' if ignore_errors else 'strict'
    with open(filename, encoding=encoding, errors=errors) as file:
        for line in file:
            yield line


def stream_csv(filename, encoding='utf-8', ignore_errors=False):
    stream = stream_lines(filename, encoding, ignore_errors)
    return csv.reader(stream)


def limit_stream(stream, count=1, skip=0):
    for i in range(skip):
        next(stream)
    for i in range(count):
        yield next(stream)


def save_obj(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('saved : {}\t{}'.format(filename, type(obj)))


def load_obj(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    logging.info('loaded : {}\t{}'.format(filename, type(obj)))
    return obj


def save_array(filename, array):
    with h5py.File(filename, 'w') as file:
        file['data'] = array
    logging.info('saved : {}\t{}\t{}'.format(filename, array.dtype, array.shape))


def load_array(filename):
    with h5py.File(filename, 'r') as file:
        array = file['data'][...]
    logging.info('loaded : {}\t{}\t{}'.format(filename, array.dtype, array.shape))
    return array


def save_sparse(filename, array):
    with h5py.File(filename, 'w') as file:
        file['shape'] = np.array(array.shape)
        file['data'] = array.data
        file['indices'] = array.indices
        file['indptr'] = array.indptr
    logging.info('saved : {}\t{}\t{}'.format(filename, array.dtype, array.shape))


def load_sparse(filename):
    with h5py.File(filename, 'r') as file:
        shape = tuple(file['shape'][...])
        data = file['data'][...]
        indices = file['indices'][...]
        indptr = file['indptr'][...]
    array = sparse.csr_matrix((data, indices, indptr), shape=shape)
    logging.info('loaded : {}\t{}\t{}'.format(filename, array.dtype, array.shape))
    return array


def save(filename, data):
    catalog = {'obj': save_obj, 'array': save_array, 'sparse': save_sparse}
    extension = filename.split('.')[-1]
    func = catalog[extension]
    func(filename, data)


def load(filename):
    catalog = {'obj': load_obj, 'array': load_array, 'sparse': load_sparse}
    extension = filename.split('.')[-1]
    func = catalog[extension]
    data = func(filename)
    return data


class Clock:

    def __init__(self):
        self.start = time.time()
        self.last = self.start
        self.now = self.start
        self.report()

    def check(self):
        self.now = time.time()
        self.report()
        self.last = self.now

    def report(self):
        txt = '\n[CLOCK]  [ {} ]    '
        txt += 'since start: [ {} ]    since last: [ {} ]\n'
        current = time.asctime().split()[3]
        since_start = datetime.timedelta(seconds=round(self.now - self.start))
        since_last = datetime.timedelta(seconds=round(self.now - self.last))
        logging.info(txt.format(current, since_start, since_last))


def beep(n=1):
    for _ in range(n):
        os.system('beep')


def print_shape_type(*objs):
    for obj in objs:
        try:
            logging.info(obj.shape, obj.dtype, type(obj))
        except AttributeError:
            logging.error(obj.shape, type(obj))


"""
def plot_maxed():
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.show()
"""
