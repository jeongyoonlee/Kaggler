from kaggler.preprocessing import TargetEncoder, EmbeddingEncoder, FrequencyEncoder
from sklearn.model_selection import KFold

from .const import RANDOM_SEED, TARGET_COL


N_FOLD = 5


def test_TargetEncoder(generate_data):
    df = generate_data()
    feature_cols = [x for x in df.columns if x != TARGET_COL]
    cat_cols = [x for x in feature_cols if df[x].nunique() < 100]

    te = TargetEncoder()
    X_cat = te.fit_transform(df[cat_cols], df[TARGET_COL])
    print('Without CV:\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(cat_cols)

    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_SEED)
    te = TargetEncoder(cv=cv)
    X_cat = te.fit_transform(df[cat_cols], df[TARGET_COL])
    print('With CV (fit_transform()):\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(cat_cols)

    te = TargetEncoder(cv=cv)
    te.fit(df[cat_cols], df[TARGET_COL])
    X_cat = te.transform(df[cat_cols])
    print('With CV (fit() and transform() separately):\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(cat_cols)


def test_EmbeddingEncoder(generate_data):
    df = generate_data()
    feature_cols = [x for x in df.columns if x != TARGET_COL]
    cat_cols = [x for x in feature_cols if df[x].nunique() < 100]
    num_cols = [x for x in feature_cols if x not in cat_cols]

    print('Test with the regression target')
    ee = EmbeddingEncoder(cat_cols=cat_cols,
                          num_cols=num_cols,
                          random_state=RANDOM_SEED)

    X_emb = ee.fit_transform(X=df[feature_cols], y=df[TARGET_COL])
    assert X_emb.shape[1] == sum(ee.n_emb)

    print('Test with the binary classification target')
    df[TARGET_COL] = (df[TARGET_COL] > df[TARGET_COL].mean()).astype(int)

    ee = EmbeddingEncoder(cat_cols=cat_cols,
                          num_cols=num_cols,
                          random_state=RANDOM_SEED)

    X_emb = ee.fit_transform(X=df[feature_cols], y=df[TARGET_COL])
    assert X_emb.shape[1] == sum(ee.n_emb)

    print('Test with the binary classification target with cross validation')
    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_SEED)
    ee = EmbeddingEncoder(cat_cols=cat_cols,
                          num_cols=num_cols,
                          cv=cv,
                          random_state=RANDOM_SEED)

    X_emb = ee.fit_transform(X=df[feature_cols], y=df[TARGET_COL])
    assert X_emb.shape[1] == sum(ee.n_emb)


def test_FrequencyEncoder(generate_data):
    df = generate_data()
    feature_cols = [x for x in df.columns if x != TARGET_COL]
    cat_cols = [x for x in feature_cols if df[x].nunique() < 100]

    te = FrequencyEncoder()
    X_cat = te.fit_transform(df[cat_cols])
    print('Without CV:\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(cat_cols)

    cv = KFold(n_splits=N_FOLD, shuffle=True, random_state=RANDOM_SEED)
    te = FrequencyEncoder(cv=cv)
    X_cat = te.fit_transform(df[cat_cols])
    print('With CV (fit_transform()):\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(cat_cols)

    te = FrequencyEncoder(cv=cv)
    te.fit(df[cat_cols])
    X_cat = te.transform(df[cat_cols])
    print('With CV (fit() and transform() separately):\n{}'.format(X_cat.head()))

    assert X_cat.shape[1] == len(cat_cols)
