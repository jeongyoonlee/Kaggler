from logging import getLogger
import numpy as np
from sklearn import base
import tensorflow as tf
from tensorflow.keras import Input, Model, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Layer, Reshape, GaussianNoise
from tensorflow.keras.losses import mean_squared_error
from tensorflow.python.keras.utils import control_flow_util

from .categorical import LabelEncoder
from .const import MIN_EMBEDDING, EMBEDDING_SUFFIX


logger = getLogger(__name__)


@tf.keras.utils.register_keras_serializable()
class BaseMasker(Layer):
    """A base class for all masking layers."""

    def __init__(self, probs, seed=[42, 33], **kwargs):
        """Initialize the layer.

        Args:
            probs (list of float): noise/masking probabilities. should have the same len as inputs.
            seed (list of float): two random seeds for two random sampling in the layer.
        """
        super().__init__(**kwargs)
        self.seed = seed
        self.probs = probs

    def call(self, inputs, training=True):
        raise NotImplementedError()

    def get_config(self):
        config = super().get_config().copy()
        config.update({'probs': self.probs,
                       'seed': self.seed})


@tf.keras.utils.register_keras_serializable()
class ZeroNoiseMasker(BaseMasker):
    """A custom Keras layer that masks inputs randomly."""

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        def mask_inputs():
            mask = tf.random.stateless_binomial(shape=tf.shape(inputs),
                                                seed=self.seed,
                                                counts=tf.ones((tf.shape(inputs)[1],)),
                                                probs=self.probs)

            return tf.where(mask == 1, tf.zeros_like(inputs), inputs)

        outputs = control_flow_util.smart_cond(training,
                                               mask_inputs,
                                               lambda: inputs)

        return outputs


@tf.keras.utils.register_keras_serializable()
class SwapNoiseMasker(BaseMasker):
    """A custom Keras layer that swaps inputs randomly."""

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


@tf.keras.utils.register_keras_serializable()
class DAELayer(Layer):
    """A DAE layer with one pair of the encoder and decoder."""

    def __init__(self, encoding_dim=128, noise_std=.0, swap_prob=.2, mask_prob=.0, random_state=42, **kwargs):
        """Initialize a DAE (Denoising AutoEncoder) layer.

        Args:
            encoding_dim (int): the numbers of hidden units in encoding/decoding layers.
            noise_std (float): standard deviation of  gaussian noise to be added to features.
            swap_prob (float): probability to add swap noise to features.
            mask_prob (float): probability to add zero masking to features.
            dropout (float): dropout probability in embedding layers
            random_state (int): random seed.
        """
        super().__init__(**kwargs)

        self.encoding_dim = encoding_dim
        self.noise_std = noise_std
        self.swap_prob = swap_prob
        self.mask_prob = mask_prob
        self.seed = random_state

        self.encoder = Dense(encoding_dim, activation='relu', name=f'{self.name}_encoder')

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.decoder = Dense(self.input_dim, activation='linear', name=f'{self.name}_decoder')

    def call(self, inputs, training):
        if training is None:
            training = K.learning_phase()

        masked_inputs = inputs
        if training:
            if self.noise_std > 0:
                masked_inputs = GaussianNoise(self.noise_std)(masked_inputs)

            if self.swap_prob > 0:
                masked_inputs = SwapNoiseMasker(probs=[self.swap_prob] * self.input_dim,
                                                seed=[self.seed] * 2)(masked_inputs)

            if self.mask_prob > 0:
                masked_inputs = ZeroNoiseMasker(probs=[self.mask_prob] * self.input_dim,
                                                seed=[self.seed] * 2)(masked_inputs)

        encoded = self.encoder(masked_inputs)
        decoded = self.decoder(encoded)

        rec_loss = K.mean(mean_squared_error(inputs, decoded))
        self.add_loss(rec_loss)

        return encoded, decoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({'encoding_dim': self.encoding_dim,
                       'noise_std': self.noise_std,
                       'swap_prob': self.swap_prob,
                       'mask_prob': self.mask_prob,
                       'random_state': self.seed})


class DAE(base.BaseEstimator):
    """Denoising AutoEncoder feature transformer."""

    def __init__(self, cat_cols=[], num_cols=[], n_emb=[], encoding_dim=128, n_layer=1, noise_std=.0,
                 swap_prob=.2, mask_prob=.0, dropout=.2, min_obs=10, n_epoch=100, batch_size=1024,
                 random_state=42):
        """Initialize a DAE (Denoising AutoEncoder) class object.

        Args:
            cat_cols (list of str): the names of categorical features to create embeddings for.
            num_cols (list of str): the names of numerical features to train embeddings with.
            n_emb (int or list of int): the numbers of embedding features used for columns.
            encoding_dim (int): the numbers of hidden units in encoding/decoding layers.
            n_layer (int): the numbers of the encoding/decoding layer pairs
            noise_std (float): standard deviation of  gaussian noise to be added to features.
            swap_prob (float): probability to add swap noise to features.
            mask_prob (float): probability to add zero masking to features.
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

        assert (encoding_dim > 0) and (n_layer > 0)
        self.encoding_dim = encoding_dim
        self.n_layer = n_layer

        assert (0. <= noise_std) and (0. <= swap_prob < 1.) and (0. <= mask_prob < 1.) and (0. <= dropout < 1.)
        self.noise_std = noise_std
        self.swap_prob = swap_prob
        self.mask_prob = mask_prob
        self.dropout = dropout

        assert (min_obs > 0) and (n_epoch > 0) and (batch_size > 0)
        self.min_obs = min_obs
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        self.seed = random_state
        self.lbe = LabelEncoder(min_obs=min_obs)

    def build_model(self, X, y=None):
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
            merged_inputs = Concatenate()(embeddings + [num_inputs]) if embeddings else num_inputs

            inputs = inputs + [num_inputs]
        else:
            merged_inputs = Concatenate()(embeddings)

        dae_layers = []
        for i in range(self.n_layer):
            dae_layers.append(DAELayer(encoding_dim=self.encoding_dim, noise_std=self.noise_std,
                                       swap_prob=self.swap_prob, mask_prob=self.mask_prob,
                                       random_state=self.seed, name=f'dae_layer_{i}'))

            encoded, decoded = dae_layers[i](merged_inputs)
            _, merged_inputs = dae_layers[i](merged_inputs, training=False)

        self.encoder = Model(inputs=inputs, outputs=encoded, name='encoder_head')
        self.dae = Model(inputs=inputs, outputs=decoded, name='decoder_head')
        self.dae.compile(optimizer='adam')

    def fit(self, X, y=None):
        """Train DAE

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series, optional): not used

        Returns:
            A trained DAE object.
        """
        if self.cat_cols:
            X[self.cat_cols] = self.lbe.fit_transform(X[self.cat_cols])

        self.build_model(X, y)

        features = [X[col].values for col in self.cat_cols]
        if self.num_cols:
            features += [X[self.num_cols].values]

        es = EarlyStopping(monitor='val_loss', min_delta=.0, patience=5, verbose=1, mode='min',
                           baseline=None, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-6, mode='min')
        self.dae.fit(x=features, y=y,
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


class SDAE(DAE):
    """Supervised Denoising AutoEncoder feature transformer."""

    def build_model(self, X, y=None):
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
            merged_inputs = Concatenate()(embeddings + [num_inputs]) if embeddings else num_inputs

            inputs = inputs + [num_inputs]
        else:
            merged_inputs = Concatenate()(embeddings)

        dae_layers = []
        for i in range(self.n_layer):
            dae_layers.append(DAELayer(encoding_dim=self.encoding_dim, noise_std=self.noise_std,
                                       swap_prob=self.swap_prob, mask_prob=self.mask_prob,
                                       random_state=self.seed, name=f'dae_layer_{i}'))

            encoded, decoded = dae_layers[i](merged_inputs)
            _, merged_inputs = dae_layers[i](merged_inputs, training=False)

        self.encoder = Model(inputs=inputs, outputs=encoded, name='encoder_head')

        if y.dtype in [np.int32, np.int64]:
            n_uniq = len(np.unique(y))
            if n_uniq == 2:
                self.n_class = 1
                self.output_activation = 'binary_crossentropy'
                self.output_loss = 'binary_crossentropy'
            elif n_uniq > 2:
                self.n_class = n_uniq
                self.output_activation = 'sparse_categorical_crossentropy'
                self.output_loss = 'sparse_categorical_crossentropy'
        else:
            self.n_class = 1
            self.output_activation = 'linear'
            self.output_loss = 'mean_squared_error'

        # supervised head
        supervised_inputs = Input((self.encoding_dim,), name='supervised_inputs')
        x = Dense(1024, 'relu')(supervised_inputs)
        x = Dropout(.3, seed=self.seed)(x)
        supervised_outputs = Dense(self.n_class, activation=self.output_activation)(x)
        self.supervised = Model(inputs=supervised_inputs, outputs=supervised_outputs, name='supervised_head')

        self.dae = Model(inputs=inputs, outputs=self.supervised(self.encoder(inputs)), name='decoder_head')
        self.dae.compile(optimizer='adam', loss=self.output_loss)

    def fit(self, X, y):
        """Train supervised DAE

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series): target variable

        Returns:
            A trained SDAE object.
        """
        assert y is not None, 'SDAE needs y (target variable) for fit()'
        super().fit(X, y)
