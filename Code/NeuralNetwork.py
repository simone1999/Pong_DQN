import numpy as np
import tensorflow.keras as keras

class DoubleDeepQAgent:
    def __init__(self, action_size, state_size, model_path, load_model=False, discount_factor=0.98, train_batch_size=32, predict_batch_size=100_000, lr=1e-3):
        self.actions = action_size  # 3
        self.stateSize = state_size  # 6
        self.discountFactor = discount_factor
        self.predictBatchSize = predict_batch_size
        self.trainBatchSize = train_batch_size
        self.lr = lr
        self.modelPath = model_path  # "../Model/good.h5"

        self.model = self.build_model()
        self.targetModel = self.build_model()

        if load_model:
            self.model = keras.models.load_model(self.modelPath)

        self.update_target_model(1.0)

    def build_model(self):
        i = keras.layers.Input(shape=(self.stateSize,))
        mask = keras.layers.Input(shape=(self.actions,))

        #x = keras.layers.Dense(128, activation="relu")(i)

        x = keras.layers.Dense(64, activation="relu")(i)

        x = keras.layers.Dense(32, activation="relu")(x)

        x = keras.layers.Dense(self.actions, activation="linear")(x)

        output = keras.layers.Multiply()([x, mask])

        model = keras.models.Model([i, mask], output)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss="logcosh")
        return model

    def predict(self, X):
        Y = self.model.predict([X, np.ones((len(X), self.actions))], batch_size=self.predictBatchSize)
        return Y

    def get_actions(self, X, randomFactor=0., excludeFirstGame=False):
        randomStart = 1 if excludeFirstGame else 0
        actions = self.predict(X)
        actionSize = len(actions[0])
        actions = np.argmax(actions, axis=-1)
        num_random = int(len(actions)/randomFactor)
        random_indexes = np.random.randint(randomStart, len(actions), size=num_random)# first Index is never random for showing Gameplay
        actions[random_indexes] = np.random.randint(0, actionSize, size=len(random_indexes))

        return actions

    def train(self, state, next_state, action, reward, done):
        #'''
        mask = np.eye(self.actions)[action]  # converts to one hot Array

        nextQTarget = self.targetModel.predict([next_state, np.ones(mask.shape)], batch_size=self.predictBatchSize)
        nextQLive = self.model.predict([next_state, np.ones(mask.shape)], batch_size=self.predictBatchSize)

        nextQTargetMax = np.max(nextQTarget, axis=-1)
        nextQLiveMax = np.max(nextQLive, axis=-1)

        nextQ = np.min([nextQTargetMax, nextQLiveMax], axis=0)

        nextQ[done] = 0
        target = reward + self.discountFactor * nextQ

        self.model.fit([state, mask], mask*target[:, None], batch_size=self.trainBatchSize, verbose=2)

        '''

        mask = np.eye(self.actions)[action]  # converts to one hot Array
        oneMask = np.ones(mask.shape)

        nextQTarget = self.targetModel.predict([next_state, oneMask], batch_size=self.predictBatchSize)
        predictedQ = self.model.predict([state, oneMask], batch_size=self.predictBatchSize)

        nextUpdatedQ = np.max(nextQTarget, axis=-1)

        nextUpdatedQ[done] = 0

        target = reward + self.discountFactor * nextUpdatedQ

        predictedQ[np.arange(len(action)), action] = target

        self.model.fit([state, oneMask], predictedQ, batch_size=self.trainBatchSize, verbose=2)

        '''

    def update_target_model(self, target_update_factor):
        model_weights = np.array(self.model.get_weights())
        target_weights = np.array(self.targetModel.get_weights())

        new_weights = target_update_factor * model_weights + (1 - target_update_factor) * target_weights

        self.targetModel.set_weights(new_weights)

    def save_model(self):
        self.model.save(self.modelPath)
