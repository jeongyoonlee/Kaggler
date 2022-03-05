import numpy as np
import pandas as pd

from kaggler.metrics import plot_curve, plot_pr_curve, plot_roc_curve

from ..const import TARGET_COL


def test_plot_curve(generate_data):
    df = generate_data()

    n_sample = df.shape[0]
    n_predictor = 2

    df[TARGET_COL] = (df[TARGET_COL] > df[TARGET_COL].mean()).astype(int)

    # Test 1-D numpy.ndarray ``p``
    plot_curve(df[TARGET_COL], np.random.rand(n_sample), name="p1", kind="roc")
    plot_curve(df[TARGET_COL], np.random.rand(n_sample), kind="pr")

    # Test 2-D numpy.ndarray ``p``
    plot_curve(
        df[TARGET_COL],
        np.random.rand(n_sample, n_predictor),
        name=["p1", "p2"],
        kind="roc",
    )
    plot_curve(df[TARGET_COL], np.random.rand(n_sample, n_predictor), kind="pr")

    # Test 1-D pandas.Series ``p``
    plot_curve(df[TARGET_COL], pd.Series(np.random.rand(n_sample)), kind="roc")
    plot_curve(df[TARGET_COL], pd.Series(np.random.rand(n_sample)), kind="pr")

    # Test 2-D pandas.DataFrame ``p``
    plot_curve(
        df[TARGET_COL], pd.DataFrame(np.random.rand(n_sample, n_predictor)), kind="roc"
    )
    plot_curve(
        df[TARGET_COL], pd.DataFrame(np.random.rand(n_sample, n_predictor)), kind="pr"
    )

    # Test 2-D list of list ``p``
    plot_curve(
        df[TARGET_COL],
        [np.random.rand(n_sample), np.random.rand(n_sample)],
        name=["p1", "p2"],
        kind="roc",
    )
    plot_curve(
        df[TARGET_COL], [np.random.rand(n_sample), np.random.rand(n_sample)], kind="pr"
    )

    # Test plot_roc_curve()
    plot_roc_curve(df[TARGET_COL], np.random.rand(n_sample))

    # Test plot_pr_curve()
    plot_pr_curve(df[TARGET_COL], np.random.rand(n_sample))
