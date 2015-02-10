# Kaggler
Kaggler is a Python package for Kaggle data science competitions and distributed under the version 3 of the GNU General Public License.

It provides online learning algorithms for classification - inspired by Kaggle user [tinrtgu's code](http://goo.gl/K8hQBx).  It uses the sparse input format that handels large sparse data efficiently.  Core code is optimized for speed by using Cython.

# Algorithms
Currently algorithms available are as follows:
* Stochastic Gradient Descent (SGD)
* Follow-the-Regularized-Leader (FTRL)
* Factorization Machine (FM)
* Neural Networks (NN) - with a single (NN) or two (NN_H2) hidden layers

# Install
## Using pip
Python package is available at PyPi for pip installation:
```
sudo pip install Kaggler
```

## From source code
If you want to install it from source code:
```
python setup.py build_ext --inplace
sudo python setup.py install
```

# Input Format
libsvm style sparse file format is used for an input.
```
1 1:1 4:1 5:0.5
0 2:1 5:1
```

# Example
## FTRL
```
from kaggler.online_model import FTRL

clf = FTRL(n=2**20,             # number of hashed features
           a=.1,                # alpha in the per-coordinate rate
           b=1,                 # beta in the per-coordinate rate
           l1=1.,               # L1 regularization parameter
           l2=1.)               # L2 regularization parameter

for x, y in clf.read_sparse('train.sparse'):
    p = clf.predict(x)          # predict for an input
    clf.update(x, p - y)        # update the model with the target using error

for x, _ in clf.read_sparse('test.sparse'):
    p = clf.predict(x)
```

## FM
```
from kaggler.online_model import FM

clf = FM(n=1e5,                 # number of features
         dim=4,                 # size of factors for interactions
         a=.01)                 # learning rate

for idx, val, y in clf.read_sparse('train.sparse'):
    p = clf.predict(idx, val)   # predict for an input
    clf.update(x, p - y)        # update the model with the target using error

for idx, val, _ in clf.read_sparse('test.sparse'):
    p = clf.predict(idx, val)
```

## NN with a single hidden layer
```
from kaggler.online_model import NN

clf = NN(n=1e5,                 # number of features
         h=16,                  # number of hidden units
         a=.1,                  # learning rate
         l2=1e-6)               # L2 regularization parameter

for idx, val, y in clf.read_sparse('train.sparse'):
    p = clf.predict(idx, val)   # predict for an input
    clf.update(x, p - y)        # update the model with the target using error

for idx, val, _ in clf.read_sparse('test.sparse'):
    p = clf.predict(idx, val)
```
