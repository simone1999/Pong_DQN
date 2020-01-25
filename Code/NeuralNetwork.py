import tensorflow.keras as keras
import numpy as np
import tensorflow as tf


class DDQN:
    def __init__(self,
                 state_shape,
                 action_size,
                 learning_rate=1e-3,
                 batch_size_predict=1024,
                 batch_size_train=128,
                 discount_factor=0.95
                 ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.batch_size_predict = batch_size_predict
        self.batch_size_train = batch_size_train
        self.discount_factor = discount_factor

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self, summary=False):
        i = keras.layers.Input(shape=self.state_shape, name="Input")

        x = keras.layers.Flatten()(i)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(self.action_size, activation=keras.activations.linear)(x)

        model = keras.models.Model(inputs=i, outputs=x)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='logcosh')

        if summary:
            model.summary()

        return model

    def train(self, states, next_states, actions, rewards, dones):

        next_q_target = self.target_model.predict(next_states, batch_size=self.batch_size_predict)
        next_q_live = self.model.predict(next_states, batch_size=self.batch_size_predict)

        next_q_target_max = np.max(next_q_target, axis=-1)
        next_q_live_max = np.max(next_q_live, axis=-1)

        next_q = np.min([next_q_target_max, next_q_live_max], axis=0)

        next_q[dones] = 0
        target = rewards + self.discount_factor * next_q

        predicted = self.model.predict(states, batch_size=self.batch_size_predict)
        mask = np.eye(self.action_size, dtype=bool)[actions]  # converts to one hot Array
        predicted[mask] = target

        self.model.fit(states, predicted, batch_size=self.batch_size_train, verbose=2)

    def get_predictions(self, states):
        return self.model.predict(states)

    def get_actions(self, states, random_factor=0.):
        q_values = self.model.predict(states)
        actions = np.argmax(q_values, axis=-1)

        random_values = np.random.uniform(0., 1., len(states))
        random_mask = random_factor > random_values
        actions[random_mask] = np.random.randint(0, self.action_size, size=np.sum(random_mask))

        return actions

    def update_target_model(self, target_update_factor):
        model_weights = np.array(self.model.get_weights())
        target_weights = np.array(self.target_model.get_weights())

        new_weights = target_update_factor * model_weights + (1. - target_update_factor) * target_weights

        self.target_model.set_weights(new_weights)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load_weights(path)
        self.target_model.set_weights(self.model.get_weights())


class AutoEncoder:
    def __init__(self, input_shape, encoding_dim, lr=1e-3, batch_size_train=32, batch_size_predict=1024):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.lr = lr
        self.batch_size_train = batch_size_train
        self.batch_size_predict = batch_size_predict

        self.encoder_decoder, self.encoder = self.build_model()

    def build_model(self, summary=False):
        encoder_in = keras.layers.Input(shape=self.input_shape)

        x = keras.layers.Flatten()(encoder_in)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        encoder_out = keras.layers.Dense(self.encoding_dim, activation=keras.activations.linear)(x)

        x = keras.layers.Dense(32, activation="relu")(encoder_out)
        x = keras.layers.Dense(64, activation="relu")(x)
        decoder_out = keras.layers.Dense(self.input_shape, activation="linear")(x)

        autoencoder = keras.models.Model(inputs=encoder_in, outputs=decoder_out)
        encoder = keras.models.Model(inputs=encoder_in, outputs=encoder_out)

        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='logcosh')

        if summary:
            autoencoder.summary()

        return autoencoder, encoder

    def train(self, x):
        self.encoder_decoder.fit(x, x, batch_size=self.batch_size_train, verbose=2)

    def encode(self, x):
        return self.encoder.predict(x, batch_size=self.batch_size_predict)


def configure_keras(num_gpus=0):
    config = tf.ConfigProto(intra_op_parallelism_threads=8,
                            inter_op_parallelism_threads=8,
                            allow_soft_placement=True,
                            device_count={'CPU': 1,
                                          'GPU': num_gpus}
                            )
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
