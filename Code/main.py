import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Code.pong import Pong
import numpy as np
import pygame
from Code.NeuralNetwork import DoubleDeepQAgent as Network
from Code.ReplayMemory import replayMemory
import math

def main():
    loadModel = False
    saveModel = True
    competiveAI = True
    numGames = 10
    modelSaveInterval = 500
    targetUpdateDelay = 100
    targetUpdatePercentage = 0.2
    batchSize = 1_000
    memorySize = 1_000_000
    randomFactor = [0, 1, 0.5, 0.5, 0.2, 0.1, 0.05, 0.01, 0, 0] #0.1
    modelPath = "../Model/test.h5"
    excludeFirstGameRandomness = True

    actionSize = 3
    stateSize = 6

    game = Pong(numGames)
    network = Network(actionSize, stateSize, modelPath, load_model=loadModel)
    memory = replayMemory(memorySize)

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
            actionRight = network.get_actions(states, randomFactor=randomFactor, excludeFirstGame=excludeFirstGameRandomness)
            if(competiveAI):
                actionLeft = network.get_actions(changeStateSide(states), randomFactor=randomFactor, excludeFirstGame=excludeFirstGameRandomness)
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
            train_states, train_next_states, train_actions, train_rewards, train_dones = memory.get_batch(batchSize)
            network.train(train_states, train_next_states, train_actions, train_rewards, train_dones)

        Q_Values = network.predict(next_states[0][np.newaxis, :])
        print(f"current Q: {np.max(Q_Values):.2}, V: {np.mean(Q_Values):.2}")

        if not iteration % targetUpdateDelay:
            print("Updating Target Model")
            network.update_target_model(targetUpdatePercentage)

        if not iteration % modelSaveInterval:
            if saveModel:
                network.save_model()

        states = next_states

        game.render()


def saveReplayMemory(memory, states, next_states, actionRight, actionLeft, rewards, dones):
    memory.save_multiple(states, next_states, actionRight, rewards[:, 1], dones)

    statesLeft = changeStateSide(states)
    next_statesLeft = changeStateSide(next_states)

    memory.save_multiple(statesLeft, next_statesLeft, actionLeft, rewards[:, 0], dones)

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
