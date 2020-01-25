import numpy as np


class ReplayMemory:
    def __init__(self, memory_length, state_size):
        self.memory_length = memory_length
        self.state_size = state_size

        self.memory_insert_index = 0
        self.memory_filled = False
        self.states = np.zeros((self.memory_length, ) + self.state_size) # State Size must be a Tuple (List) eg. (5, ) or (20, 15)
        self.next_states = np.zeros((self.memory_length, ) + self.state_size)
        self.actions = np.zeros(self.memory_length, np.int)
        self.rewards = np.zeros(self.memory_length)
        self.dones = np.zeros(self.memory_length, np.bool)

    def append(self, states, next_states, actions, rewards, dones):
        assert len(states) == len(next_states) == len(actions) == len(rewards) == len(dones)
        insertion_length = len(actions)
        start_index = self.memory_insert_index
        end_index = start_index + insertion_length
        wrapping_end_index = end_index
        insert_length = insertion_length
        if end_index > self.memory_length:
            wrapping_end_index = self.memory_length
            insert_length = wrapping_end_index - start_index
        self.states[start_index:wrapping_end_index] = states[:insert_length]
        self.next_states[start_index:wrapping_end_index] = next_states[:insert_length]
        self.actions[start_index:wrapping_end_index] = actions[:insert_length]
        self.rewards[start_index:wrapping_end_index] = rewards[:insert_length]
        self.dones[start_index:wrapping_end_index] = dones[:insert_length]
        if end_index > self.memory_length:
            self.memory_filled = True
            wrapping_end_index = end_index % self.memory_length
            insert_length = wrapping_end_index
            self.states[:wrapping_end_index] = states[-wrapping_end_index:]
            self.next_states[:wrapping_end_index] = next_states[-wrapping_end_index:]
            self.actions[:wrapping_end_index] = actions[-wrapping_end_index:]
            self.rewards[:wrapping_end_index] = rewards[-wrapping_end_index:]
            self.dones[:wrapping_end_index] = dones[-wrapping_end_index:]
        self.memory_insert_index = wrapping_end_index

    def get_data(self, num_samples, weighted=False):
        memory_length = self.memory_length if self.memory_filled else self.memory_insert_index
        if not weighted:
            indexes = np.random.randint(0, memory_length, size=num_samples)
        else:
            weights = np.zeros(memory_length)
            unique_rewards = np.unique(self.rewards[:memory_length])
            occourances = []
            for reward in unique_rewards:
                mask = self.rewards[:memory_length] == reward
                occourance = np.sum(mask)
                occourances.append(occourance)
                weighting = (1 / len(unique_rewards)) / occourance
                weights[mask] = weighting
            assert np.sum(weights == 0) == 0
            print([(x, y) for x, y in zip(unique_rewards, occourances)])
            indexes = np.random.choice(np.arange(memory_length), num_samples, p=weights)
        return (
            self.states[indexes],
            self.next_states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.dones[indexes],
        )
