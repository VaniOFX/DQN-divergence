import argparse
import random
from collections import deque
from copy import deepcopy
from typing import List

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam


def linear_epsilon_anneal(it):
    return max(1 - (it * 0.95/1000), 0.05)
    return max(1 - (it * 0.95/100000), 0.05)


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_ordered_samples(self, batch_size):
        return list(self.memory)[-batch_size:]

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, hidden_sizes, device):
        super(DQN, self).__init__()
        model = nn.ModuleList()

        for size_tuple in zip(hidden_sizes, hidden_sizes[1:-1]):
            model.extend([nn.Linear(*size_tuple), nn.ReLU()])

        model.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        self.model = nn.Sequential(*model).to(device)

    def forward(self, input):
        return self.model(input)


class DQNAgent:
    def __init__(self, action_size, hidden_sizes, memory_capacity=2000, epsilon=1,
                 discount_factor=0.8, learning_rate=1e-3, use_target_net=False, update_target_freq=10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_size = action_size
        self.memory = ReplayMemory(memory_capacity)
        self.discount_factor = discount_factor  # discount rate
        self.epsilon = epsilon                  # exploration rate
        self.Q_model = DQN(hidden_sizes=hidden_sizes, device=self.device)
        self.Q_model_target = deepcopy(self.Q_model)
        self.optimizer = Adam(self.Q_model.parameters(), lr=learning_rate)
        self.max_q_val = torch.tensor(0.0).to(self.device)
        self.use_target_net = use_target_net
        self.update_target_freq = update_target_freq

    def memorize(self, state, action, reward, next_state, done):
        state       = torch.from_numpy(state).float().to(self.device)
        action      = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward      = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state  = torch.from_numpy(next_state).float().to(self.device)
        done        = torch.tensor(done, dtype=torch.bool).to(self.device)

        self.memory.push((state, action, reward, next_state, done))

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def sample_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            return self.Q_model(state)[0].argmax().item()

    def compute_q_vals(self, states, actions):
        Q_values = self.Q_model(states)
        result = torch.gather(Q_values, 1, actions)

        self.max_q_val = torch.max(self.max_q_val, result.abs().max())

        return result

    def get_max_q_val(self):
        return self.max_q_val

    def compute_targets(self, batch_size, rewards, next_states, dones):
        target_model = self.Q_model_target if self.use_target_net else self.Q_model

        next_states = next_states.view(batch_size, -1)

        Q_values = target_model(next_states).max(dim=1, keepdim=True)[0]
        Q_values.masked_fill_(dones, 0)

        return rewards + self.discount_factor * Q_values

    def train(self, batch_size, sample_memory=False):
        transitions = self.memory.sample(batch_size) if sample_memory\
            else self.memory.get_ordered_samples(batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.stack(states)
        actions = torch.stack(actions)[:, None]  # Need 64 bit to use them as index
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)[:, None]
        dones = torch.stack(dones)[:, None]  # Boolean

        # combine dimensions in case of multi-dim input (except batch dim)
        states = states.view(batch_size, -1)

        # clipping of rewards
        rewards[rewards > 1.0] = 1.0
        rewards[rewards < -1.0] = -1.0

        q_val = self.compute_q_vals(states, actions)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = self.compute_targets(batch_size, rewards, next_states, dones)

        loss = F.smooth_l1_loss(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self, global_updates):
        if self.use_target_net and global_updates % self.update_target_freq == 0:
            self.Q_model_target = deepcopy(self.Q_model)

    def load_checkpoint(self, name):
        self.Q_model = torch.load(name, map_location=self.device)

    def save_checkpoint(self, name):
        torch.save(self.Q_model, name)


if __name__ == "__main__":
    # and it's still not reproducible...
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="CartPole-v1", type=str, help='the environment to run experiments on')
    parser.add_argument('--batch_size', default=32, type=int, help='the batch size of DQN')
    parser.add_argument('--sample_memory', default=True, type=bool, help='whether to use memory replay or correlated samples')
    parser.add_argument('--hidden_sizes', default=[128], type=List[int], help='size of DQN hidden layers')
    parser.add_argument('--num_episodes', default=200, type=int, help='number of episodes to run DQN')
    parser.add_argument('--epsilon', default=1.0, type=float, help='the change ot picking a random action')
    parser.add_argument('--discount_factor', default=0.8, type=float, help='how much to discount future rewards')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of DQN')
    parser.add_argument('--use_target_net', default=True, type=bool, help='whether to use target network')
    parser.add_argument('--update_target_freq', default=10000, type=int, help='how frequently to update target net')

    args = parser.parse_args()

    env = gym.envs.make(args.env)

    print(f"Training on '{args.env}'")

    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    agent = DQNAgent(action_size=action_size,
                     hidden_sizes=[state_size, *args.hidden_sizes, action_size],
                     epsilon=args.epsilon,
                     discount_factor=args.discount_factor,
                     learning_rate=args.lr,
                     use_target_net=args.use_target_net,
                     update_target_freq=args.update_target_freq)
    print(agent.Q_model)

    batch_size = args.batch_size
    global_steps = 0
    average_steps = []

    for episode in range(args.num_episodes):
        state = env.reset()

        steps = 0
        while True:
            agent.set_epsilon(linear_epsilon_anneal(global_steps))
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            steps += 1

            if done:
                average_steps.append(steps)
                if episode % 20 == 0:
                # if episode % 1 == 0:
                    print(f"Episode: {episode:05d}/{args.num_episodes:05d}\t\t "
                          f"Score: {np.mean(average_steps):.1f}\t\t "
                          f"Max-|Q|: {agent.get_max_q_val():.1f}\t\t "
                          f"Epsilon: {agent.epsilon:.2f}")
                    average_steps = []
                break

            if len(agent.memory) > batch_size:
                agent.train(batch_size, sample_memory=args.sample_memory)
                agent.update_target_network(global_steps)
                global_steps += 1
