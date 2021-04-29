[![PyPI version](https://badge.fury.io/py/Kaggler.svg)](https://badge.fury.io/py/Kaggler)
[![Build Status](https://travis-ci.org/jeongyoonlee/Kaggler.svg?branch=master)](https://travis-ci.org/jeongyoonlee/Kaggler)
[![Downloads](https://pepy.tech/badge/kaggler)](https://pepy.tech/project/kaggler)
[![codecov](https://codecov.io/gh/jeongyoonlee/Kaggler/branch/master/graph/badge.svg)](https://codecov.io/gh/jeongyoonlee/Kaggler)


# Kaggler
Kaggler is a Python package for lightweight online machine learning algorithms and utility functions for ETL and data analysis. It is distributed under the MIT License.

Its online learning algorithms are inspired by Kaggle user [tinrtgu's code](http://goo.gl/K8hQBx).  It uses the sparse input format that handles large sparse data efficiently.  Core code is optimized for speed by using Cython.


## Installation

### Dependencies
Python packages required are listed in `requirements.txt`
* cython
* h5py
* hyperopt
* lightgbm
* ml_metrics
* numpy/scipy
* pandas
* scikit-learn

### Using pip
Python package is available at PyPi for pip installation:
```
pip install -U Kaggler
```
If installation fails because it cannot find `MurmurHash3.h`, please add `.` to
`LD_LIBRARY_PATH` as described [here](https://github.com/jeongyoonlee/Kaggler/issues/32).

### From source code
If you want to install it from source code:
```
python setup.py build_ext --inplace
python setup.py install
```


## Feature Engineering

### One-Hot, Label, Target, Frequency, and Embedding Encoders for Categorical Features
```python
import pandas as pd
from kaggler.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder, FrequencyEncoder, EmbeddingEncoder

trn = pd.read_csv('train.csv')
target_col = trn.columns[-1]
cat_cols = [col for col in trn.columns if trn[col].dtype == 'object']

ohe = OneHotEncoder(min_obs=100) # grouping all categories with less than 100 occurences
lbe = LabelEncoder(min_obs=100)  # grouping all categories with less than 100 occurences
te = TargetEncoder()			 # replacing each category with the average target value of the category
fe = FrequencyEncoder()	         # replacing each category with the frequency value of the category
ee = EmbeddingEncoder()          # mapping each category to a vector of real numbers

X_ohe = ohe.fit_transform(trn[cat_cols])	    # X_ohe is a scipy sparse matrix
trn[cat_cols] = lbe.fit_transform(trn[cat_cols])
trn[cat_cols] = te.fit_transform(trn[cat_cols])
trn[cat_cols] = fe.fit_transform(trn[cat_cols])
X_ee = ee.fit_transform(trn[cat_cols])          # X_ee is a numpy matrix

tst = pd.read_csv('test.csv')
X_ohe = ohe.transform(tst[cat_cols])
tst[cat_cols] = lbe.transform(tst[cat_cols])
tst[cat_cols] = te.transform(tst[cat_cols])
tst[cat_cols] = fe.transform(tst[cat_cols])
X_ee = ee.transform(tst[cat_cols])
```

### Denoising AutoEncoder (DAE)
```python
import pandas as pd
from kaggler.preprocessing import DAE

trn = pd.read_csv('train.csv')
tst = pd.read_csv('test.csv')
target_col = trn.columns[-1]
cat_cols = [col for col in trn.columns if trn[col].dtype == 'object']
num_cols = [col for col in trn.columns if col not in cat_cols + [target_col]]

dae = DAE(cat_cols=cat_cols, num_cols=num_cols, n_encoding=128)
X = dae.fit_transform(pd.concat([trn, tst], axis=0))    # encoding input features into the encoding vectors with size of 128
```

## AutoML

### Feature Selection & Hyperparameter Tuning
```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from kaggler.metrics import auc
from kaggler.model import AutoLGB


RANDOM_SEED = 42
N_OBS = 10000
N_FEATURE = 100
N_IMP_FEATURE = 20

X, y = make_classification(n_samples=N_OBS,
                            n_features=N_FEATURE,
                            n_informative=N_IMP_FEATURE,
                            random_state=RANDOM_SEED)
X = pd.DataFrame(X, columns=['x{}'.format(i) for i in range(X.shape[1])])
y = pd.Series(y)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                                test_size=.2,
                                                random_state=RANDOM_SEED)

model = AutoLGB(objective='binary', metric='auc')
model.tune(X_trn, y_trn)
model.fit(X_trn, y_trn)
p = model.predict(X_tst)
print('AUC: {:.4f}'.format(auc(y_tst, p)))

```

## Ensemble

### Netflix Blending
```python
import numpy as np
from kaggler.ensemble import netflix
from kaggler.metrics import rmse

# Load the predictions of input models for ensemble
p1 = np.loadtxt('model1_prediction.txt')
p2 = np.loadtxt('model2_prediction.txt')
p3 = np.loadtxt('model3_prediction.txt')

# Calculate RMSEs of model predictions and all-zero prediction.
# At a competition, RMSEs (or RMLSEs) of submissions can be used.
y = np.loadtxt('target.txt')
e0 = rmse(y, np.zeros_like(y))
e1 = rmse(y, p1)
e2 = rmse(y, p2)
e3 = rmse(y, p3)

p, w = netflix([e1, e2, e3], [p1, p2, p3], e0, l=0.0001) # l is an optional regularization parameter.
```


## Algorithms
Currently algorithms available are as follows:

### Online learning algorithms
* Stochastic Gradient Descent (SGD)
* Follow-the-Regularized-Leader (FTRL)
* Factorization Machine (FM)
* Neural Networks (NN) - with a single (NN) or two (NN_H2) ReLU hidden layers
* Decision Tree

### Batch learning algorithm
* Neural Networks (NN) - with a single hidden layer and L-BFGS optimization

### Examples
```python
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
from kaggler import load_data

X, y = load_data('train.sps')

clf.fit(X, y)
p = clf.predict(X)
```

## Data I/O
Kaggler supports CSV (`.csv`), LibSVM (`.sps`), and HDF5 (`.h5`) file formats:
```
# CSV format: target,feature1,feature2,...
1,1,0,0,1,0.5
0,0,1,0,0,5

# LibSVM format: target feature-index1:feature-value1 feature-index2:feature-value2
1 1:1 4:1 5:0.5
0 2:1 5:1

# HDF5
- issparse: binary flag indicating whether it stores sparse data or not.
- target: stores a target variable as a numpy.array
- shape: available only if issparse == 1. shape of scipy.sparse.csr_matrix
- indices: available only if issparse == 1. indices of scipy.sparse.csr_matrix
- indptr: available only if issparse == 1. indptr of scipy.sparse.csr_matrix
- data: dense feature matrix if issparse == 0 else data of scipy.sparse.csr_matrix
```

```python
from kaggler.data_io import load_data, save_data

X, y = load_data('train.csv')	# use the first column as a target variable
X, y = load_data('train.h5')	# load the feature matrix and target vector from a HDF5 file.
X, y = load_data('train.sps')	# load the feature matrix and target vector from LibSVM file.

save_data(X, y, 'train.csv')
save_data(X, y, 'train.h5')
save_data(X, y, 'train.sps')
```

## Documentation
Package documentation is available at [here](https://kaggler.readthedocs.io/en/latest/)
