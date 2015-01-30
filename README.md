# Kaggler
Kaggler is a Python package for Kaggle data science competitions and distributed under the version 3 of the GNU General Public License.

It provides online learning algorithms for classification and regression.  It uses the sparse input format that handels large sparse data efficiently.  Core code is optimized for speed by using Cython.

# Algorithms
Currently algorithms available are as follows:
* Follow-the-Regularized-Leader (FTRL) - implementation inspired by Kaggle user tinrtgu's code at http://goo.gl/K8hQBx
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

