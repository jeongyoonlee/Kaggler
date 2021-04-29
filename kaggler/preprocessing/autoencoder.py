from logging import getLogger
import numpy as np
from sklearn import base
import tensorflow as tf
from tensorflow.keras import Input, Model, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Layer, Reshape
from tensorflow.python.keras.utils import control_flow_util

from .categorical import LabelEncoder
from .const import MIN_EMBEDDING, EMBEDDING_SUFFIX


logger = getLogger(__name__)


@tf.keras.utils.register_keras_serializable()
class SwapNoiseMasker(Layer):
    """A custom Keras layer that swaps inputs randomly."""
    def __init__(self, probs, seed=[42, 33], **kwargs):
        """Initialize the layer.

        Args:
            probs (list of float): swap noise probabilities. should have the same len as inputs.
            seed (list of float): two random seeds for two random sampling in the layer.
        """
        super().__init__(**kwargs)
        self.seed = seed
        self.probs = probs

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def mask_inputs():
            mask = tf.random.stateless_binomial(shape=tf.shape(inputs),
                                                seed=self.seed,
                                                counts=tf.ones((tf.shape(inputs)[1],)),
                                                probs=self.probs)

            # tf.random.shuffle() without tf.gather() doesn't work in a custom layer
            # ref: https://github.com/tensorflow/tensorflow/issues/6269#issuecomment-465850464
            return tf.where(mask == 1,
                            tf.gather(inputs, tf.random.shuffle(tf.range(tf.shape(inputs)[0]), seed=self.seed[0])),
                            inputs)

        outputs = control_flow_util.smart_cond(training,
                                               mask_inputs,
                                               lambda: inputs)

        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({'probs': self.probs,
                       'seed': self.seed})
        return config


class DAE(base.BaseEstimator):
    """Denoising AutoEncoder feature transformer."""

    def __init__(self, cat_cols=[], num_cols=[], n_emb=[], encoding_dim=128, noise_prob=.2,
                 dropout=.2, min_obs=10, n_epoch=20, batch_size=1024, random_state=42):
        """Initialize a DAE (Denoising AutoEncoder) class object.

        Args:
            cat_cols (list of str): the names of categorical features to create embeddings for.
            num_cols (list of str): the names of numerical features to train embeddings with.
            n_emb (int or list of int): the numbers of embedding features used for columns.
            noise_prob (float): probability to add swap noise to features.
            dropout (float): dropout probability in embedding layers
            min_obs (int): categories observed less than it will be grouped together before training embeddings
            n_epoch (int): the number of epochs to train a neural network with embedding layer
            batch_size (int): the size of mini-batches in model training
            random_state (int): random seed.
        """
        assert cat_cols or num_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        if isinstance(n_emb, int):
            self.n_emb = [n_emb] * len(cat_cols)
        elif isinstance(n_emb, list):
            if not n_emb:
                self.n_emb = [None] * len(cat_cols)
            else:
                assert len(cat_cols) == len(n_emb)
                self.n_emb = n_emb
        else:
            raise ValueError('n_emb should be int or list')

        assert encoding_dim > 0
        self.encoding_dim = encoding_dim

        assert (0. <= noise_prob < 1.) and (0. <= dropout < 1.)
        self.noise_prob = noise_prob
        self.dropout = dropout

        assert (min_obs > 0) and (n_epoch > 0) and (batch_size > 0)
        self.min_obs = min_obs
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        self.seed = random_state
        self.lbe = LabelEncoder(min_obs=min_obs)

    def build_model(self, X):
        inputs = []
        num_inputs = []
        embeddings = []

        if self.cat_cols:
            for i, col in enumerate(self.cat_cols):
                n_uniq = X[col].nunique()
                if not self.n_emb[i]:
                    self.n_emb[i] = max(MIN_EMBEDDING, 2 * int(np.log2(n_uniq)))

                inp = Input(shape=(1,), name=col)
                emb = Embedding(input_dim=n_uniq, output_dim=self.n_emb[i], name=col + EMBEDDING_SUFFIX)(inp)
                emb = Dropout(self.dropout)(emb)
                emb = Reshape((self.n_emb[i],))(emb)

                inputs.append(inp)
                embeddings.append(emb)

        if self.num_cols:
            num_inputs = Input(shape=(len(self.num_cols),), name='num_inputs')
            merged_inputs = Concatenate()(embeddings + [num_inputs])

            inputs = inputs + [num_inputs]
        else:
            merged_inputs = Concatenate()(embeddings)

        input_dim = sum(self.n_emb) + len(self.num_cols)

        masked_inputs = SwapNoiseMasker(probs=[self.noise_prob] * input_dim, seed=[self.seed] * 2)(merged_inputs)
        encoded = Dense(self.encoding_dim, activation='relu', name='encoder')(masked_inputs)
        decoded = Dense(input_dim, activation='linear', name='decoder')(encoded)

        self.encoder = Model(inputs=inputs, outputs=encoded)
        self.dae = Model(inputs=inputs, outputs=decoded)

        self.dae.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, X, y=None):
        """Train DAE

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series, optional): not used

        Returns:
            A trained AutoEncoder object.
        """
        if self.cat_cols:
            X[self.cat_cols] = self.lbe.fit_transform(X[self.cat_cols])

        self.build_model(X)

        features = [X[col].values for col in self.cat_cols]
        if self.num_cols:
            features += [X[self.num_cols].values]

        es = EarlyStopping(monitor='val_loss', min_delta=.001, patience=5, verbose=1, mode='min',
                           baseline=None, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-6, mode='min')
        self.dae.fit(x=features, y=features,
                     epochs=self.n_epoch,
                     validation_split=.2,
                     batch_size=self.batch_size,
                     callbacks=[es, rlr])

    def transform(self, X):
        """Encode features using the DAE trained

        Args:
            X (pandas.DataFrame): features to encode

        Returns:
            Encoding matrix for features
        """
        if self.cat_cols:
            X[self.cat_cols] = self.lbe.transform(X[self.cat_cols])

        features = [X[col].values for col in self.cat_cols]
        if self.num_cols:
            features += [X[self.num_cols].values]

        return self.encoder.predict(features)

    def fit_transform(self, X, y=None):
        """Train DAE and encode features using the DAE trained

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series, optional): not used

        Returns:
            Encoding matrix for features
        """
        self.fit(X, y)
        return self.transform(X)
