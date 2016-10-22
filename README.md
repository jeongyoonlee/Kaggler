# Kaggler
Kaggler is a Python package for Kaggle data science competitions and distributed under the version 3 of the GNU General Public License.

It provides online learning algorithms for classification - inspired by Kaggle user [tinrtgu's code](http://goo.gl/K8hQBx).  It uses the sparse input format that handles large sparse data efficiently.  Core code is optimized for speed by using Cython.

# Algorithms
Currently algorithms available are as follows:

## Online learning algorithms
* Stochastic Gradient Descent (SGD)
* Follow-the-Regularized-Leader (FTRL)
* Factorization Machine (FM)
* Neural Networks (NN) - with a single (NN) or two (NN_H2) ReLU hidden layers
* Decision Tree

## Batch learning algorithm
* Neural Networks (NN) - with a single hidden layer and L-BFGS optimization

# Dependencies
Python packages required are listed in `requirements.txt`
* cython
* h5py
* numpy/scipy
* pandas
* scikit-learn
* ml_metrics

# Installation
## Using pip
Python package is available at PyPi for pip installation:
```
sudo pip install -U Kaggler
```

## From source code
If you want to install it from source code:
```
python setup.py build_ext --inplace
sudo python setup.py install
```

# Input Format
libsvm style sparse file format is used.
```
1 1:1 4:1 5:0.5
0 2:1 5:1
```

# Example
```
from kaggler.online_model import SGD, FTRL, FM, NN

# SGD
clf = SGD(a=.01,                # learning rate
          l1=1e-6,              # L1 regularization parameter
          l2=1e-6,              # L2 regularization parameter
          n=2**20,              # number of hashed features
          epoch=10,             # number of epochs
          interaction=True)     # use feature interaction or not

# FTRL
clf = FTRL(a=.1,                # alpha in the per-coordinate rate
           b=1,                 # beta in the per-coordinate rate
           l1=1.,               # L1 regularization parameter
           l2=1.,               # L2 regularization parameter
           n=2**20,             # number of hashed features
           epoch=1,             # number of epochs
           interaction=True)    # use feature interaction or not

# FM
clf = FM(n=1e5,                 # number of features
         epoch=100,             # number of epochs
         dim=4,                 # size of factors for interactions
         a=.01)                 # learning rate

# NN
clf = NN(n=1e5,                 # number of features
         epoch=10,              # number of epochs
         h=16,                  # number of hidden units
         a=.1,                  # learning rate
         l2=1e-6)               # L2 regularization parameter

# online training and prediction directly with a libsvm file
for x, y in clf.read_sparse('train.sparse'):
    p = clf.predict_one(x)      # predict for an input
    clf.update_one(x, p - y)    # update the model with the target using error

for x, _ in clf.read_sparse('test.sparse'):
    p = clf.predict_one(x)

# online training and prediction with a scipy sparse matrix
from sklearn.datasets import load_svmlight_file

X, y = load_svmlight_file('train.sparse')

clf.fit(X, y)
p = clf.predict(X)
```

# Package Documentation
Package documentation is available at [here](http://pythonhosted.org//Kaggler).
