# Kaggler
Kaggler is a Python package for Kaggle data science competitions and distributed under the version 3 of the GNU General Public License.

It provides online learning algorithms for classification - inspired by Kaggle user [tinrtgu's code](http://goo.gl/K8hQBx).  It uses the sparse input format that handels large sparse data efficiently.  Core code is optimized for speed by using Cython.

# Algorithms
Currently algorithms available are as follows:
* Stochastic Gradient Descent (SGD)
* Follow-the-Regularized-Leader (FTRL)
* Factorization Machine (FM)
* Neural Networks (NN)

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
```
from kaggler.online_model import FTRL   # FTRL algorithm

clf = FTRL(a=.1, b=1, l1=1., l2=1.)
for x, y in clf.read_sparse('train.sparse'):
    p = clf.predict(x)      # predict for an input
    clf.update(x, p - y)    # update the model with the target using error

for x, _ in clf.read_sparse('test.sparse'):
    p = clf.predict(x)
```
