from collections import deque
import random
import numpy as np


class replayMemory:
    def __init__(self, length):
        self.memory = deque(maxlen=length)

    def save_single(self, state, next_state, action, reward, done):
        self.memory.append([state, next_state, action, reward, done])

    def save_multiple(self, states, next_states, actions, rewards, dones):
        for state, next_state, action, reward, done in zip(states, next_states, actions, rewards, dones):
            self.save_single(state, next_state, action, reward, done)

    def get_batch(self, batchSize):
        assert batchSize <= len(self.memory)
        batch = random.sample(self.memory, batchSize)
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        for state, next_state, action, reward, done in batch:
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return states, next_states, actions, rewards, dones


