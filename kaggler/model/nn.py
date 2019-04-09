from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import sparse
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

import logging
import numpy as np
import time

from ..const import SEC_PER_MIN


logger = logging.getLogger('Kaggler')


class NN(object):
    """Implement a neural network with a single h layer."""

    def __init__(self, n=5, h=10, b=100000, l1=.0, l2=.0, random_state=None):
        """Initialize the NN class object.

        Args:
            h (int): number of h nodes
            b (int): number of input examples to be processed together to find
                     the second order gradient for back-propagation
            n (int): number of epoches
            l1 (float): regularization parameter for weights between the input
                        and hidden layers
            l2 (float): regularization parameter for weights between the hidden
                        and output layers.
        """

        np.random.seed(random_state)
        self.h = h
        self.b = b
        self.n = n
        self.l1 = l1
        self.l2 = l2
        self.n_opt = 0

    def fit(self, X, y, X_val=None, y_val=None):
        """Train a network with the quasi-Newton method.
        
        Args:
            X (np.array of float): feature matrix for training
            y (np.array of float): target values for training
            X_val (np.array of float): feature matrix for validation
            y_val (np.array of float): target values for validation
        """
        y = y.reshape((len(y), 1))

        if sparse.issparse(X):
            X = X.tocsr()

        if X_val is not None:
            n_val = len(y_val)
            y_val = y_val.reshape((n_val, 1))

        # Set initial weights randomly.
        self.i = X.shape[1]
        self.l1 = self.l1 / self.i
        self.w = (np.random.rand((self.i + 2) * self.h + 1) - .5) * 1e-6
        self.w_opt = self.w
        self.n_opt = 0

        logger.info('training ...')
        n_obs = X.shape[0]
        batch = self.b
        n_epoch = self.n
        idx = range(n_obs)
        self.auc_opt = .5

        start = time.time()
        print('\tEPOCH TRAIN     VALID     BEST      TIME (m)')
        print('\t--------------------------------------------')

        # Before training
        p = self.predict_raw(X)
        auc = roc_auc_score(y, p)
        auc_val = auc
        if X_val is not None:
            p_val = self.predict_raw(X_val)
            auc_val = roc_auc_score(y_val, p_val)

        print('\t{:3d}:  {:.6f}  {:.6f}  {:.6f}  {:.2f}'.format(
              0, auc, auc_val, self.auc_opt,
              (time.time() - start) / SEC_PER_MIN))
     
        # Use 'while' instead of 'for' to increase n_epoch if the validation
        # error keeps improving at the end of n_epoch 
        epoch = 1
        while epoch <= n_epoch:
            # Shuffle inputs every epoch - it helps avoiding the local optimum
            # when batch < n_obs.
            np.random.shuffle(idx)

            # Find the optimal weights for batch input examples.
            # If batch == 1, it's the stochastic optimization, which is slow
            # but uses minimal memory.  If batch == n_obs, it's the batch
            # optimization, which is fast but uses maximum memory.
            # Otherwise, it's the mini-batch optimization, which balances the
            # speed and space trade-offs.
            for i in range(int(n_obs / batch) + 1):
                if (i + 1) * batch > n_obs:
                    sub_idx = idx[batch * i:n_obs]
                else:
                    sub_idx = idx[batch * i:batch * (i + 1)]

                x = X[sub_idx]
                neg_idx = [n_idx for n_idx, n_y in enumerate(y[sub_idx]) if n_y == 0.]
                pos_idx = [p_idx for p_idx, p_y in enumerate(y[sub_idx]) if p_y == 1.]
                x0 = x[neg_idx]
                x1 = x[pos_idx]
                # Update weights to minimize the cost function using the
                # quasi-Newton method (L-BFGS-B), where:
                #   func -- cost function
                #   jac -- jacobian (derivative of the cost function)
                #   maxiter -- number of iterations for L-BFGS-B
                ret = minimize(self.func,
                               self.w,
                               args=(x0, x1),
                               method='L-BFGS-B',
                               jac=self.fprime,
                               options={'maxiter': 5})
                self.w = ret.x

            p = self.predict_raw(X)
            auc = roc_auc_score(y, p)
            auc_val = auc

            if X_val is not None:
                p_val = self.predict_raw(X_val)
                auc_val = roc_auc_score(y_val, p_val)

                if auc_val > self.auc_opt:
                    self.auc_opt = auc_val
                    self.w_opt = self.w
                    self.n_opt = epoch

                    # If validation auc is still improving after n_epoch,
                    # try 10 more epochs
                    if epoch == n_epoch:
                        n_epoch += 5

            print('\t{:3d}:  {:.6f}  {:.6f}  {:.6f}  {:.2f}'.format(
                  epoch, auc, auc_val, self.auc_opt,
                  (time.time() - start) / SEC_PER_MIN))

            epoch += 1

        if X_val is not None:
            print('Optimal epoch is {0} ({1:.6f})'.format(self.n_opt,
                                                          self.auc_opt))
            self.w = self.w_opt

        logger.info('done training')

    def predict(self, X):
        """Predict targets for a feature matrix.

        Args:
            X (np.array of float): feature matrix for prediction

        Returns:
            prediction (np.array)
        """
        logger.info('predicting ...')
        ps = self.predict_raw(X)

        return sigm(ps[:, 0])

    def predict_raw(self, X):
        """Predict targets for a feature matrix.

        Args:
            X (np.array of float): feature matrix for prediction
        """
        # b -- bias for the input and h layers
        b = np.ones((X.shape[0], 1))
        w2 = self.w[-(self.h + 1):].reshape(self.h + 1, 1)
        w1 = self.w[:-(self.h + 1)].reshape(self.i + 1, self.h)

        # Make X to have the same number of columns as self.i.
        # Because of the sparse matrix representation, X for prediction can
        # have a different number of columns.
        if X.shape[1] > self.i:
            # If X has more columns, cut extra columns.
            X = X[:, :self.i]
        elif X.shape[1] < self.i:
            # If X has less columns, cut the rows of the weight matrix between
            # the input and h layers instead of X itself because the SciPy
            # sparse matrix does not support .set_shape() yet.
            idx = range(X.shape[1])
            idx.append(self.i)        # Include the last row for the bias
            w1 = w1[idx, :]

        if sparse.issparse(X):
            return np.hstack((sigm(sparse.hstack((X, b)).dot(w1)), b)).dot(w2)
        else:
            return np.hstack((sigm(np.hstack((X, b)).dot(w1)), b)).dot(w2)

    def func(self, w, *args):
        """Return the costs of the neural network for predictions.

        Args:
            w (array of float): weight vectors such that:
                w[:-h1] -- weights between the input and h layers
                w[-h1:] -- weights between the h and output layers
            args: features (args[0]) and target (args[1])

        Returns:
            combined cost of RMSE, L1, and L2 regularization
        """
        x0 = args[0]
        x1 = args[1]

        n0 = x0.shape[0]
        n1 = x1.shape[0]

        # n -- number of pairs to evaluate
        n = max(n0, n1) * 10
        idx0 = np.random.choice(range(n0), size=n)
        idx1 = np.random.choice(range(n1), size=n)

        # b -- bias for the input and h layers
        b0 = np.ones((n0, 1))
        b1 = np.ones((n1, 1))
        i1 = self.i + 1
        h = self.h
        h1 = h + 1

        # Predict for features -- cannot use predict_raw() because here
        # different weights can be used.
        if sparse.issparse(x0):
            p0 = np.hstack((sigm(sparse.hstack((x0, b0)).dot(w[:-h1].reshape(
                               i1, h))), b0)).dot(w[-h1:].reshape(h1, 1))
            p1 = np.hstack((sigm(sparse.hstack((x1, b1)).dot(w[:-h1].reshape(
                               i1, h))), b1)).dot(w[-h1:].reshape(h1, 1))
        else:
            p0 = np.hstack((sigm(np.hstack((x0, b0)).dot(w[:-h1].reshape(
                               i1, h))), b0)).dot(w[-h1:].reshape(h1, 1))
            p1 = np.hstack((sigm(np.hstack((x1, b1)).dot(w[:-h1].reshape(
                               i1, h))), b1)).dot(w[-h1:].reshape(h1, 1))

        p0 = p0[idx0]
        p1 = p1[idx1]

        # Return the cost that consists of the sum of squared error +
        # L2-regularization for weights between the input and h layers +
        # L2-regularization for weights between the h and output layers.
        #return .5 * (sum((1 - sigm(p1 - p0)) ** 2) + self.l1 * sum(w[:-h1] ** 2) +
        return .5 * (sum((1 - p1 + p0) ** 2) / n +
                     self.l1 * sum(w[:-h1] ** 2) / (i1 * h) +
                     self.l2 * sum(w[-h1:] ** 2) / h1)

    def fprime(self, w, *args):
        """Return the derivatives of the cost function for predictions.

        Args:
            w (array of float): weight vectors such that:
                w[:-h1] -- weights between the input and h layers
                w[-h1:] -- weights between the h and output layers
            args: features (args[0]) and target (args[1])

        Returns:
            gradients of the cost function for predictions
        """

        x0 = args[0]
        x1 = args[1]

        n0 = x0.shape[0]
        n1 = x1.shape[0]

        # n -- number of pairs to evaluate
        n = max(n0, n1) * 10
        idx0 = np.random.choice(range(n0), size=n)
        idx1 = np.random.choice(range(n1), size=n)

        # b -- bias for the input and h layers
        b = np.ones((n, 1))
        i1 = self.i + 1
        h = self.h
        h1 = h + 1

        w2 = w[-h1:].reshape(h1, 1)
        w1 = w[:-h1].reshape(i1, h)

        if sparse.issparse(x0):
            x0 = x0.tocsr()[idx0]
            x1 = x1.tocsr()[idx1]
            xb0 = sparse.hstack((x0, b))
            xb1 = sparse.hstack((x1, b))
        else:
            x0 = x0[idx0]
            x1 = x1[idx1]
            xb0 = np.hstack((x0, b))
            xb1 = np.hstack((x1, b))

        z0 = np.hstack((sigm(xb0.dot(w1)), b))
        z1 = np.hstack((sigm(xb1.dot(w1)), b))
        y0 = z0.dot(w2)
        y1 = z1.dot(w2)

        #e = 1 - sigm(y1 - y0)
        #dy = e * dsigm(y1 - y0)
        e = 1 - (y1 - y0)
        dy = e / n

        # Calculate the derivative of the cost function w.r.t. F and w2 where:
        # F -- weights between the input and h layers
        # w2 -- weights between the h and output layers
        dw1 = -(xb1.T.dot(dy.dot(w2[:-1].reshape(1, h)) * dsigm(xb1.dot(w1))) -
               xb0.T.dot(dy.dot(w2[:-1].reshape(1, h)) * dsigm(xb0.dot(w1)))
                       ).reshape(i1 * h) + self.l1 * w[:-h1] / (i1 * h)
        dw2 = -(z1 - z0).T.dot(dy).reshape(h1) + self.l2 * w[-h1:] / h1

        return np.append(dw1, dw2)


def sigm(x):
    """Return the value of the sigmoid function at x.

    Args:
        x (np.array of float or float)

    Returns:
        value(s) of the sigmoid function for x.
    """

    # Avoid numerical overflow by capping the input to the exponential
    # function - doesn't affect the return value.
    return 1 / (1 + np.exp(-np.maximum(x, -20)))


def dsigm(x):
    """Return the value of derivative of sigmoid function w.r.t. x.
    Args:
        x (np.array of float or float)

    Returns:
        derivative(s) of the sigmoid function w.r.t. x.
    """

    return sigm(x) * (1 - sigm(x))
