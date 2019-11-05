from Code.pong import Pong
import numpy as np
import time
from tqdm import tqdm
from Code.NeuralNetwork import Network as Network
from Code.ReplayMemory import replayMemory

def main():
    numGames = 10
    targetUpdateDelay = 100
    train_start = 1_000
    batchSize = 500
    memorySize = 100_000
    numActions = 3
    randomFactor = 0.1
    excludeFirstGameRandomness = True

    game = Pong(numGames)
    network = Network()
    memory = replayMemory(memorySize)
    states, _, _ = game.step(np.zeros(shape=numGames), np.zeros(shape=numGames))
    #for iteration in tqdm(range(1_000_000)):
    for iteration in range(1_000_000_000):
        actionLeft = np.zeros(shape=numGames)
        #actionRight = np.zeros(shape=numGames)

        verticalSpeed = states[:, 3]
        nextY = states[:, 1] + verticalSpeed / 10

        actionLeft[nextY > states[:, 4] + 0.02] = 1
        actionLeft[nextY < states[:, 4] - 0.02] = 2

        #actionRight[nextY > states[:, 5] + 0.05] = 1
        #actionRight[nextY < states[:, 5] - 0.05] = 2

        if iteration > train_start:
            actionRight = network.getActions(states, randomFactor=randomFactor, excludeFirstGame=excludeFirstGameRandomness)
        else:
            actionRight = np.random.randint(0, numActions, size=len(states))

        next_states, rewards, dones = game.step(actionLeft, actionRight)

        rewards[np.invert(dones)] += 0.05

        memory.save_multiple(states, next_states, actionRight, rewards[:, 1], dones)

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
            network.updateTargetModel()

        states = next_states

        game.render()
        #time.sleep(0.01)

main()