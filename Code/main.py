import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Code.pong import Pong
import numpy as np
import pygame
from Code.NeuralNetwork import DDQN
from Code.NeuralNetwork import AutoEncoder
from Code.ReplayMemory import ReplayMemory
import math

def main():
    loadModel = False
    saveModel = True
    competiveAI = True
    numGames = 1
    modelSaveInterval = 5_000
    targetUpdateDelay = 1_000
    targetUpdatePercentage = 0.2
    batchSize = 1_000
    memorySize = 1_000_000
    randomFactor = 0.25
    modelPath = "../Model/test.h5"

    actionSize = 3
    stateSize = 6
    encodingDim = 10

    game = Pong(numGames)
    memory = ReplayMemory(memorySize, (stateSize,))
    auto_encoder = AutoEncoder(stateSize, encodingDim)
    network = DDQN(encodingDim, actionSize)

    if loadModel:
        network.load_model(path=modelPath)

    train_start = math.ceil(batchSize / numGames)
    autoplay = True
    previousKeyADown = False
    iteration = 0

    states, _, _ = game.step(np.zeros(shape=numGames), np.zeros(shape=numGames))

    while True:
        iteration += 1
        print(f"Iteration: {iteration}")
        keys = pygame.key.get_pressed()
        if (keys[pygame.K_a] and not previousKeyADown):
            autoplay = not autoplay
        previousKeyADown = keys[pygame.K_a]

        if iteration > train_start:
            encoded = auto_encoder.encode(states)
            actionRight = network.get_actions(encoded)
            if(competiveAI):
                encoded = auto_encoder.encode(changeStateSide(states))
                actionLeft = network.get_actions(encoded, random_factor=randomFactor)
            else:
                actionLeft = simpleBotLeft(numGames, states)
        else:
            actionRight = np.random.randint(0, actionSize, size=numGames)
            actionLeft = simpleBotLeft(numGames, states)

        if not autoplay:
            if(keys[pygame.K_DOWN]):
                actionLeft[0] = 1
            elif(keys[pygame.K_UP]):
                actionLeft[0] = 2
            else:
                actionLeft[0] = 0

        next_states, rewards, dones = game.step(actionLeft, actionRight)

        # rewards[np.invert(dones)] += 0.01

        saveReplayMemory(memory, states, next_states, actionRight, actionLeft, rewards, dones)

        if iteration > train_start:
            train_states, train_next_states, train_actions, train_rewards, train_dones = memory.get_data(batchSize)
            auto_encoder.train(train_states)

            encoded = auto_encoder.encode(train_states)
            encoded_next = auto_encoder.encode(train_next_states)
            network.train(encoded, encoded_next, train_actions, train_rewards, train_dones)

        encoded = auto_encoder.encode(next_states[0][np.newaxis, :])
        Q_Values = network.get_predictions(encoded)
        print(f"current Q: {np.max(Q_Values):.2}, V: {np.mean(Q_Values):.2}")

        if not iteration % targetUpdateDelay:
            print("Updating Target Model")
            network.update_target_model(targetUpdatePercentage)

        if not iteration % modelSaveInterval:
            if saveModel:
                network.save_model(modelPath)

        states = next_states

        game.render()


def saveReplayMemory(memory, states, next_states, actionRight, actionLeft, rewards, dones):
    memory.append(states, next_states, actionRight, rewards[:, 1], dones)


    statesLeft = changeStateSide(states)
    next_statesLeft = changeStateSide(next_states)

    memory.append(statesLeft, next_statesLeft, actionLeft, rewards[:, 0], dones)


def changeStateSide(states):
    states = states.copy()
    states[:, 0] *= -1
    states[:, 2] *= -1

    tmp = states[:, 4].copy()
    states[:, 4] = states[:, 5]
    states[:, 5] = tmp
    return states


def simpleBotLeft(numGames, states):
    actionLeft = np.zeros(shape=numGames, dtype=int)

    ballY = states[:, 1]

    actionLeft[ballY > states[:, 4] + 0.02] = 1
    actionLeft[ballY < states[:, 4] - 0.02] = 2

    return actionLeft


main()
