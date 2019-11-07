import numpy as np
import tensorflow.keras as keras

class Network:
    def __init__(self, loadModel):
        self.actions = 3
        self.stateSize = 6
        self.discountFactor = 0.98
        self.predictBatchSize = 100_000
        self.trainBatchSize = 32
        self.lr = 1e-3

        self.model = self.buildModel()
        self.targetModel = self.buildModel()

        if loadModel:
            self.model = keras.models.load_model("../Model/good.h5")

        self.updateTargetModel(1.0)


    def buildModel(self):
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

    def getActions(self, X, randomFactor=0., excludeFirstGame=False):
        randomStart = 1 if excludeFirstGame else 0
        actions = self.predict(X)
        actionSize = len(actions[0])
        actions = np.argmax(actions, axis=-1)
        num_random = int(len(actions)/randomFactor)
        random_indexes = np.random.randint(randomStart, len(actions), size=num_random)# first Index is never random for showing Gameplay
        actions[random_indexes] = np.random.randint(0, actionSize, size=len(random_indexes))

        return actions

    def train(self, state, nextState, action, reward, done):
        #'''
        mask = np.eye(self.actions)[action]  # converts to one hot Array

        nextQTarget = self.targetModel.predict([nextState, np.ones(mask.shape)], batch_size=self.predictBatchSize)
        nextQLive = self.model.predict([nextState, np.ones(mask.shape)], batch_size=self.predictBatchSize)

        nextQTargetMax = np.max(nextQTarget, axis=-1)
        nextQLiveMax = np.max(nextQLive, axis=-1)

        nextQ = np.min([nextQTargetMax, nextQLiveMax], axis=0)

        nextQ[done] = 0
        target = reward + self.discountFactor * nextQ

        self.model.fit([state, mask], mask*target[:, None], batch_size=self.trainBatchSize, verbose=2)

        '''

        mask = np.eye(self.actions)[action]  # converts to one hot Array
        oneMask = np.ones(mask.shape)

        nextQTarget = self.targetModel.predict([nextState, oneMask], batch_size=self.predictBatchSize)
        predictedQ = self.model.predict([state, oneMask], batch_size=self.predictBatchSize)

        nextUpdatedQ = np.max(nextQTarget, axis=-1)

        nextUpdatedQ[done] = 0

        target = reward + self.discountFactor * nextUpdatedQ

        predictedQ[np.arange(len(action)), action] = target

        self.model.fit([state, oneMask], predictedQ, batch_size=self.trainBatchSize, verbose=2)

        '''


    def updateTargetModel(self, targetUpdateFactor):
        modelWeights = np.array(self.model.get_weights())
        targetWeights = np.array(self.targetModel.get_weights())

        newWeights = targetUpdateFactor * modelWeights + (1 - targetUpdateFactor) * targetWeights

        self.targetModel.set_weights(newWeights)

