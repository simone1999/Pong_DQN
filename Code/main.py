from Code.pong import Pong
import numpy as np
import pygame
import time
from tqdm import tqdm
from Code.NeuralNetwork import Network as Network
from Code.ReplayMemory import replayMemory

def main():
    loadModel = True
    competiveAI = True
    numGames = 10
    targetUpdateDelay = 100
    targetUpdateFactor = 0.1
    train_start = 1_000
    batchSize = 500
    memorySize = 100_000
    numActions = 3
    randomFactor = 0.5
    excludeFirstGameRandomness = True

    game = Pong(numGames)
    network = Network(loadModel)
    memory = replayMemory(memorySize)
    states, _, _ = game.step(np.zeros(shape=numGames), np.zeros(shape=numGames))
    autoplay = True
    previousKeyADown = False
    #for iteration in tqdm(range(1_000_000)):
    for iteration in range(1_000_000_000):
        keys = pygame.key.get_pressed()
        if (keys[pygame.K_a] and not previousKeyADown):
            autoplay = not autoplay
        previousKeyADown = keys[pygame.K_a]

        #actionRight[nextY > states[:, 5] + 0.05] = 1
        #actionRight[nextY < states[:, 5] - 0.05] = 2

        if iteration > train_start:
            actionRight = network.getActions(states, randomFactor=randomFactor, excludeFirstGame=excludeFirstGameRandomness)
            if(competiveAI):
                actionLeft = network.getActions(changeStateSide(states), randomFactor=randomFactor, excludeFirstGame=excludeFirstGameRandomness)
            else:
                actionLeft = simpleBotLeft(numGames, states)
        else:
            actionRight = np.random.randint(0, numActions, size=len(states))
            actionLeft = simpleBotLeft(numGames, states)

        if not autoplay:
            if(keys[pygame.K_DOWN]):
                actionLeft[0] = 1
            elif(keys[pygame.K_UP]):
                actionLeft[0] = 2
            else:
                actionLeft[0] = 0

        next_states, rewards, dones = game.step(actionLeft, actionRight)

        rewards[np.invert(dones)] += 0.01

        saveReplayMemory(memory, states, next_states, actionRight, actionLeft, rewards, dones)
        #memory.save_multiple(states, next_states, actionRight, rewards[:, 1], dones)


        if iteration > train_start:
            train_states, train_next_states, train_actions, train_rewards, train_dones = memory.get_batch(batchSize)
            network.train(train_states, train_next_states, train_actions, train_rewards, train_dones)

        print(iteration)

        Q_Values = network.predict(next_states[0][np.newaxis, :])
        Q_Value = np.sum(Q_Values)
        print(f"current Q Value: {Q_Value}")
        print(f"current Reward: {rewards[0]}")

        if not iteration % targetUpdateDelay:
            print("Updating Target Model")
            network.updateTargetModel(targetUpdateFactor)

        states = next_states

        game.render()
        #time.sleep(0.01)


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
    # actionRight = np.zeros(shape=numGames, dtype=int)

    verticalSpeed = states[:, 3]
    nextY = states[:, 1] + verticalSpeed / 10

    actionLeft[nextY > states[:, 4] + 0.02] = 1
    actionLeft[nextY < states[:, 4] - 0.02] = 2

    return actionLeft




main()