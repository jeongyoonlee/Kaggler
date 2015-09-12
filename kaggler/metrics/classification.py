from __future__ import division
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import log_loss


def logloss(y, p):
    """Bounded log loss error.
    
    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        bounded log loss error
    """

    p[p < 1e-15] = 1e-15
    p[p > 1 - 1e-15] = 1 - 1e-15
    return log_loss(y, p)
