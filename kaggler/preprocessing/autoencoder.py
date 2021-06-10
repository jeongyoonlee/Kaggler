from logging import getLogger
import numpy as np
from sklearn import base
from sklearn.utils import check_random_state
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

    def __init__(self, encoding_dim=128, n_encoder=1, noise_std=.0, swap_prob=.2, mask_prob=.0, seed=42,
                 **kwargs):
        """Initialize a DAE (Denoising AutoEncoder) layer.

        Args:
            encoding_dim (int): the numbers of hidden units in encoding/decoding layers.
            n_encoder (int): the numbers of hidden encoding layers.
            noise_std (float): standard deviation of  gaussian noise to be added to features.
            swap_prob (float): probability to add swap noise to features.
            mask_prob (float): probability to add zero masking to features.
            dropout (float): dropout probability in embedding layers
            seed (int): random seed.
        """
        super().__init__(**kwargs)

        self.encoding_dim = encoding_dim
        self.n_encoder = n_encoder
        self.noise_std = noise_std
        self.swap_prob = swap_prob
        self.mask_prob = mask_prob
        self.seed = seed

        self.encoders = [Dense(encoding_dim, activation='relu', name=f'{self.name}_encoder_{i}')
                         for i in range(self.n_encoder)]

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

        x = masked_inputs
        encoded_list = []
        for encoder in self.encoders:
            x = encoder(x)
            encoded_list.append(x)

        encoded = Concatenate()(encoded_list) if len(encoded_list) > 1 else encoded_list[0]
        decoded = self.decoder(encoded)

        rec_loss = K.mean(mean_squared_error(inputs, decoded))
        self.add_loss(rec_loss)

        return encoded, decoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({'encoding_dim': self.encoding_dim,
                       'n_encoder': self.n_encoder,
                       'noise_std': self.noise_std,
                       'swap_prob': self.swap_prob,
                       'mask_prob': self.mask_prob,
                       'random_state': self.seed})


class DAE(base.BaseEstimator):
    """Denoising AutoEncoder feature transformer."""

    def __init__(self, cat_cols=[], num_cols=[], n_emb=[], encoding_dim=128, n_layer=1, n_encoder=1,
                 noise_std=.0, swap_prob=.2, mask_prob=.0, dropout=.2, min_obs=1, n_epoch=10, batch_size=1024,
                 random_state=42, label_encoding=False):
        """Initialize a DAE (Denoising AutoEncoder) class object.

        Args:
            cat_cols (list of str): the names of categorical features to create embeddings for.
            num_cols (list of str): the names of numerical features to train embeddings with.
            n_emb (int or list of int): the numbers of embedding features used for columns.
            encoding_dim (int): the numbers of hidden units in encoding/decoding layers.
            n_layer (int): the numbers of the encoding/decoding layer pairs
            n_encoder (int): the numbers of encoding layers in each of the encoding/decoding pairs
            noise_std (float): standard deviation of  gaussian noise to be added to features.
            swap_prob (float): probability to add swap noise to features.
            mask_prob (float): probability to add zero masking to features.
            dropout (float): dropout probability in embedding layers
            min_obs (int): categories observed less than it will be grouped together before training embeddings
            n_epoch (int): the number of epochs to train a neural network with embedding layer
            batch_size (int): the size of mini-batches in model training
            random_state (int or np.RandomState): random seed.
            label_encoding (bool): to label-encode categorical columns (True) or not (False)
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

        assert (encoding_dim > 0) and (n_layer > 0) and (n_encoder > 0)
        self.encoding_dim = encoding_dim
        self.n_layer = n_layer
        self.n_encoder = n_encoder

        assert (0. <= noise_std) and (0. <= swap_prob < 1.) and (0. <= mask_prob < 1.) and (0. <= dropout < 1.)
        self.noise_std = noise_std
        self.swap_prob = swap_prob
        self.mask_prob = mask_prob
        self.dropout = dropout

        assert (min_obs > 0) and (n_epoch > 0) and (batch_size > 0)
        self.min_obs = min_obs
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        # Following Scikit-Learn's coding guidelines (https://scikit-learn.org/stable/developers/develop.html):
        # 1. Every keyword argument accepted by __init__ should correspond to an attribute on the instance.
        # 2. The routine should accept a keyword random_state and use this to construct a np.random.RandomState
        #    object
        self.random_state = random_state
        self.random_state_ = check_random_state(self.random_state)

        # Get an integer seed from np.random.RandomState to use it for tensorflow
        self.seed = self.random_state_.get_state()[1][0]
        self.label_encoding = label_encoding
        if self.label_encoding:
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
            merged_inputs = Concatenate()(embeddings) if len(embeddings) > 1 else embeddings[0]

        dae_layers = []
        for i in range(self.n_layer):
            dae_layers.append(DAELayer(encoding_dim=self.encoding_dim, n_encoder=self.n_encoder,
                                       noise_std=self.noise_std, swap_prob=self.swap_prob, mask_prob=self.mask_prob,
                                       seed=self.seed, name=f'dae_layer_{i}'))

            encoded, decoded = dae_layers[i](merged_inputs)
            _, merged_inputs = dae_layers[i](merged_inputs, training=False)

        self.encoder = Model(inputs=inputs, outputs=encoded, name='encoder_model')
        self.dae = Model(inputs=inputs, outputs=decoded, name='decoder_model')
        self.dae.compile(optimizer='adam')

    def fit(self, X, y=None, validation_data=None):
        """Train DAE

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series, optional): not used
            validation_data (list of pandas.DataFrame and pandas.Series): validation features and target

        Returns:
            A trained DAE object.
        """
        if validation_data is not None:
            if y is None:
                X_val = validation_data[0]
                y_val = None
            else:
                X_val, y_val = validation_data

        if self.cat_cols and self.label_encoding:
            X[self.cat_cols] = self.lbe.fit_transform(X[self.cat_cols])

            if validation_data is not None:
                X_val[self.cat_cols] = self.lbe.fit_transform(X_val[self.cat_cols])

        self.build_model(X, y)

        features = [X[col].values for col in self.cat_cols]
        if self.num_cols:
            features += [X[self.num_cols].values]

        if validation_data is not None:
            features_val = [X_val[col].values for col in self.cat_cols]
            if self.num_cols:
                features_val += [X_val[self.num_cols].values]

        es = EarlyStopping(monitor='val_loss', min_delta=.0, patience=5, verbose=1, mode='min',
                           baseline=None, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-6, mode='min')
        if validation_data is None:
            self.dae.fit(x=features, y=y,
                         epochs=self.n_epoch,
                         validation_split=.2,
                         batch_size=self.batch_size,
                         callbacks=[es, rlr])
        else:
            self.dae.fit(x=features, y=y,
                         epochs=self.n_epoch,
                         validation_data=(features_val, y_val),
                         batch_size=self.batch_size,
                         callbacks=[es, rlr])

    def transform(self, X):
        """Encode features using the DAE trained

        Args:
            X (pandas.DataFrame): features to encode

        Returns:
            Encoding matrix for features
        """
        X = X.copy()
        if self.cat_cols and self.label_encoding:
            X[self.cat_cols] = self.lbe.transform(X[self.cat_cols])

        features = [X[col].values for col in self.cat_cols]
        if self.num_cols:
            features += [X[self.num_cols].values]

        return self.encoder.predict(features)

    def fit_transform(self, X, y=None, validation_data=None):
        """Train DAE and encode features using the DAE trained

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series, optional): not used
            validation_data (list of pandas.DataFrame and pandas.Series): validation features and target

        Returns:
            Encoding matrix for features
        """
        self.fit(X, y, validation_data)
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
            merged_inputs = Concatenate()(embeddings) if len(embeddings) > 1 else embeddings[0]

        dae_layers = []
        for i in range(self.n_layer):
            dae_layers.append(DAELayer(encoding_dim=self.encoding_dim, n_encoder=self.n_encoder,
                                       noise_std=self.noise_std, swap_prob=self.swap_prob, mask_prob=self.mask_prob,
                                       seed=self.seed, name=f'dae_layer_{i}'))

            encoded, decoded = dae_layers[i](merged_inputs)
            _, merged_inputs = dae_layers[i](merged_inputs, training=False)

        self.encoder = Model(inputs=inputs, outputs=encoded, name='encoder_model')
        self.decoder = Model(inputs=inputs, outputs=decoded, name='decoder_model')

        if y.dtype in [np.int32, np.int64]:
            n_uniq = len(np.unique(y))
            if n_uniq == 2:
                self.n_class = 1
                self.output_activation = 'sigmoid'
                self.output_loss = 'binary_crossentropy'
            elif n_uniq > 2:
                self.n_class = n_uniq
                self.output_activation = 'softmax'
                self.output_loss = 'sparse_categorical_crossentropy'
        else:
            self.n_class = 1
            self.output_activation = 'linear'
            self.output_loss = 'mean_squared_error'

        # supervised head
        x = Dense(1024, 'relu')(encoded)
        x = Dropout(.3)(x)
        supervised_outputs = Dense(self.n_class, activation=self.output_activation)(x)

        self.dae = Model(inputs=inputs, outputs=supervised_outputs, name='supervised_model')
        self.dae.compile(optimizer='adam', loss=self.output_loss)

    def fit(self, X, y, validation_data=None):
        """Train supervised DAE

        Args:
            X (pandas.DataFrame): features to encode
            y (pandas.Series): target variable
            validation_data (list of pandas.DataFrame and pandas.Series): validation features and target

        Returns:
            A trained SDAE object.
        """
        assert y is not None, 'SDAE needs y (target variable) for fit()'
        super().fit(X, y, validation_data)
