import torch
import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ['state','action','reward','next_state','done','mask','log_prob'])

class ReplayMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, mask, log_prob):
        if not torch.is_tensor(state['adj_matrix']):
            state['adj_matrix'] = torch.tensor(state['adj_matrix'])
        if not torch.is_tensor(next_state['adj_matrix']):
            next_state['adj_matrix'] = torch.tensor(next_state['adj_matrix'])
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.bool)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(state, action, reward, next_state, done, mask, log_prob)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = [t.state for t in batch]
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = [t.next_state for t in batch]
        dones = np.array([t.done for t in batch], dtype=np.float32)
        masks = np.array([t.mask for t in batch], dtype=bool)
        log_probs = np.array([t.log_prob for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones, masks, log_probs

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.position = 0

class DiffusionMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def append(self, state, action_onehot):
        self.memory.append((state, action_onehot))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, action_onehots = zip(*batch)
        action_onehots = np.stack(action_onehots).astype(np.float32)
        action_onehots = torch.from_numpy(action_onehots).to('cuda' if torch.cuda.is_available() else 'cpu')
        return list(states), action_onehots, None

    def __len__(self):
        return len(self.memory)

class HighRewardTrajectoryMemory:
    def __init__(self, capacity: int, action_dim: int):
        self.capacity = capacity
        self.episodes = []
        self.min_heap = []
        self.action_dim = action_dim
        self.next_idx = 0

    def getlength(self):
        return sum(len(episode[0]) for episode in self.episodes)

    def __len__(self):
        return len(self.episodes)

    def add_trajectory(self, states_list, actions_list, total_reward):
        if len(self.episodes) < self.capacity:
            idx = self.next_idx
            self.episodes.append((states_list, actions_list, total_reward))
            random.shuffle(self.episodes)  # simple priority maintenance
            self.next_idx += 1
        else:
            min_reward = min(self.episodes, key=lambda x: x[2])[2]
            if total_reward > min_reward:
                self.episodes.append((states_list, actions_list, total_reward))
                self.episodes.sort(key=lambda x: x[2], reverse=True)
                self.episodes = self.episodes[:self.capacity]

    def sample(self, batch_size):
        states_batch = []
        actions_batch = []
        for _ in range(batch_size):
            ep_idx = random.randrange(0, len(self.episodes))
            states_list, actions_list, _ = self.episodes[ep_idx]
            t = random.randrange(0, len(states_list))
            state_t = states_list[t]
            action_t = actions_list[t]
            states_batch.append(state_t)
            action_onehot = np.zeros(self.action_dim, dtype=np.float32)
            action_onehot[action_t] = 1.0
            actions_batch.append(action_onehot)
        action_onehots = np.stack(actions_batch).astype(np.float32)
        return states_batch, action_onehots, None
