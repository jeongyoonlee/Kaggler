import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_roc_curve(y, p):
    """Plot a ROC (receiver operating characteristic) curve for the binary classification label ``y`` and predictions ``p``.

    Args:
        y (pd.Series or np.ndarray): A vector of binary classification label [0, 1]
        p (pd.DataFrame, pd.Series or np.ndarray): One or multiple vectors of classification predictions

    Returns:
        None
    """

    if isinstance(p, pd.DataFrame):
        for col in p.columns:
            fpr, tpr, _ = roc_curve(y, p[col].values)
            plt.plot(fpr, tpr, label=col)
    else:
        name = p.name if isinstance(p, pd.Series) else "Prediction"
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()


def plot_pr_curve(y, p):
    """Plot a precision-recall curve for the binary classification label ``y`` and predictions ``p``.

    Args:
        y (pd.Series or np.ndarray): A vector of binary classification label [0, 1]
        p (pd.DataFrame, pd.Series or np.ndarray): One or multiple vectors of classification predictions

    Returns:
        None
    """

    if isinstance(p, pd.DataFrame):
        for col in p.columns:
            precision, recall, _ = precision_recall_curve(y, p[col].values)
            plt.step(recall, precision, where="post", label=col)
    else:
        name = p.name if isinstance(p, pd.Series) else "Prediction"
        precision, recall, _ = precision_recall_curve(y, p)
        plt.step(recall, precision, where="post", label=name)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()


plot_registry = {"roc": plot_roc_curve, "pr": plot_pr_curve}


def plot_curve(y, p, name=None, kind="roc"):
    """Plot a metric curve (e.g. a ROC or PR curve) for the binary classification label ``y`` and predictions ``p``.

    Args:
        y (pd.Series or np.ndarray): A vector of binary classification label [0, 1]
        p (pd.DataFrame, pd.Series or np.ndarray): One or multiple vectors of classification predictions
        name (str or list of str): The name(s) of the predictions. Used in the legend of the plot
        kind (str): A curve type to plot. Use "roc" for a ROC curve or "pr" for a PR curve.

    Returns:
        None
    """
    assert isinstance(
        p, (pd.Series, pd.DataFrame, np.ndarray, list)
    ), f"Invalid type, {type(p)} for p"

    # Convert the prediction input, ``p`` into pd.DataFrame with ``name`` as column names
    if isinstance(p, list):
        p = np.array(p)
        if p.ndim == 2:
            p = p.T

    if len(p.shape) == 1 or p.shape[1] == 1:
        if isinstance(p, pd.Series):
            p = pd.DataFrame(p)
        else:
            name = name if name is not None else "Prediction"
            p = pd.DataFrame({name: p})

    elif isinstance(p, np.ndarray):
        assert name is None or (
            isinstance(name, list) and len(name) == p.shape[1]
        ), f"Invalid name, {name} for {p.shape[1]} prediction(s)"
        name = (
            name
            if name is not None
            else [f"Prediction {i + 1}" for i in range(p.shape[1])]
        )
        p = pd.DataFrame({name[i]: p[:, i] for i in range(p.shape[1])})

    assert isinstance(p, pd.DataFrame) and len(p.shape) == 2
    assert p.shape[0] == len(y)

    assert kind in plot_registry.keys(), f"Invalid kind, {kind}"
    plot_registry[kind](y, p)
