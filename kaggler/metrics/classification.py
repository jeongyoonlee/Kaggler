import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, log_loss
from sklearn.metrics import roc_auc_score as auc            # noqa 
from sklearn.metrics import average_precision_score as ap   # noqa

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


plot_registry = {
    "auc": plot_roc_curve,
    "pr": plot_pr_curve
}


def plot_curve(y, p, name=None, metric="auc"):
    assert isinstance(p, (pd.Series, pd.DataFrame, np.ndarray, list)), f"Invalid type, {type(p)} for p"

    # Convert the prediction input, ``p`` into pd.DataFrame with ``name`` as column names
    if isinstance(p, list):
        p = np.array(p)
        
    if len(p.shape) == 1 or p.shape[1] == 1:
        if isinstance(p, pd.Series):
            p = pd.DataFrame(p)
        else:
            name = name if name is not None else "Prediction"
            p = pd.DataFrame({name: p})

    elif isinstance(p, np.ndarray):
        assert name is None or (isinstance(name, list) and len(name) == p.shape[1]), f"Invalid name, {name}"
        name = name if name is not None else [f"Prediction {i + 1}" for i in range(p.shape[1])]
        p = pd.DataFrame({name[i]: p[:, i] for i in range(p.shape[1])})

    assert isinstance(p, pd.DataFrame) and len(p.shape) == 2
    assert p.shape[0] == len(y)
    
    assert metric in plot_registry.keys(), f"Invalid metric, {metric}"
    registry[metric](y, p)
    

def plot_roc_curve(y, p):
    if isinstance(p, pd.DataFrame):
        for col in p.columns:
            fpr, tpr, _ = roc_curve(y, p[col].values)
            plt.plot(fpr, tpr, label=col)
    else:
        name = p.name if isinstance(p, pd.Series) else "Prediction"
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=name)
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    
def plot_pr_curve(y, p):
    if isinstance(p, pd.DataFrame):
        for col in p.columns:
            precision, recall, _ = precision_recall_curve(y, p[col].values)
            plt.step(recall, precision, where='post', label=col)
    else:
        name = p.name if isinstance(p, pd.Series) else "Prediction"
        precision, recall, _ = precision_recall_curve(y, p[col].values)
        plt.step(recall, precision, where='post', label=col)
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
