from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import roc_curve, precision_recall_curve, log_loss, roc_auc_score as auc
import matplotlib.pyplot as plt

from ..const import EPS


def logloss(y, p):
    """Bounded log loss error.

    Args:
        y (numpy.array): target
        p (numpy.array): prediction

    Returns:
        bounded log loss error
    """

    p[p < EPS] = EPS
    p[p > 1 - EPS] = 1 - EPS
    return log_loss(y, p)


def plot_roc_curve(y, p):
    fpr, tpr, _ = roc_curve(y, p)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


def plot_pr_curve(y, p):
    precision, recall, _ = precision_recall_curve(y, p)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
