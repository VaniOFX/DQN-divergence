import os
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from copy import deepcopy
from typing import List
import hydra
from hydra.core.config_store import ConfigStore

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim


def linear_epsilon_anneal(it, exploration_steps):
    # the paper vary the epsilon linearly from 1.0 to 0.1
    # before reaching the lower bound of 0.1, they ran it for 1/50 million frames
    # also, they seem to run 50K steps with a completely random policy, before
    # moving a epsilon-greedy policy (with an initial eps of 1.0)
    # NOTE: during evaluation, the authors of DQN changes this to 0.05 fixed (without exploration)
    lower_bound = 0.1

    return max(1 - (it * (1.0-lower_bound)/exploration_steps), lower_bound)


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
                 discount_factor=0.99, optimizer='Adam', learning_rate=1e-3,
                 use_target_net=False, update_target_freq=10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_size = action_size
        self.memory = ReplayMemory(memory_capacity)
        self.discount_factor = discount_factor  # discount rate
        self.epsilon = epsilon                  # exploration rate
        self.Q_model = DQN(hidden_sizes=hidden_sizes, device=self.device)

        self.Q_model_target = deepcopy(self.Q_model)
        self.optimizer = self.build_optimizer(optimizer, learning_rate)
        self.max_q_val = torch.tensor(0.0).to(self.device)
        self.use_target_net = use_target_net
        self.update_target_freq = update_target_freq

        self.clear_max_q_val()

    def build_optimizer(self, name, learning_rate):
        # perhaps not the best way, but gives good flexibility
        cls = getattr(torch.optim, name)

        return cls(self.Q_model.parameters(), lr=learning_rate)

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

    def clear_max_q_val(self):
        self.max_q_val = torch.tensor(0.0).to(self.device)

    def get_max_q_val(self):
        return self.max_q_val

    def compute_targets(self, batch_size, rewards, next_states, dones):
        target_model = self.Q_model_target if self.use_target_net else self.Q_model

        next_states = next_states.view(batch_size, -1)

        Q_values = target_model(next_states).max(dim=1, keepdim=True)[0]
        Q_values.masked_fill_(dones, 0)

        return rewards + self.discount_factor * Q_values

    def train(self, batch_size, sample_memory=False):
        # from the paper:
        # > A more sophisticated sampling strat-egy might emphasize transitions
        # > from which we can learn the most, similar toprioritized sweeping
        # perhaps we can look into this?
        # see how it affects divergence?

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
        rewards = torch.clamp(rewards, -1.0, 1.0)

        q_val = self.compute_q_vals(states, actions)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = self.compute_targets(batch_size, rewards, next_states, dones)

        # loss = F.smooth_l1_loss(q_val, target)
        loss = F.l1_loss(q_val, target, reduction='none')
        loss = loss.clamp(-1.0, 1.0).mean()

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


@dataclass
class RLConfig:
    seed: int = field(default=42, metadata="random seed for the environments")
    env: str = field(default="CartPole-v1", metadata="the environment to run experiments on")
    batch_size: int = field(default=32, metadata="the batch size of to train DQN")
    sample_memory: bool =field(default=True, metadata="whether to use memory replay or correlated samples")
    memory_capacity: int =field(default=1000000, metadata='Memory capacity for experience replay')
    hidden_sizes: List[int] = field(default_factory=lambda: [128], metadata="list of hidden layer dimensions for DQN")
    num_episodes: int = field(default=200, metadata="number of episodes to run DQN for")
    epsilon: float = field(default=1.0, metadata="the change ot picking a random action")
    discount_factor: float =field(default=0.99, metadata="discount over future rewards")
    exploration_steps: int =field(default=1500, metadata='Number of steps before the eps-greedy policy reaches its optima')
    optimizer: str =field(default='Adam', metadata='Optimizer to use')
    lr: float = field(default=1e-3, metadata="learning rate to train DQN")
    use_target_net: bool = field(default=True, metadata="whether to use target network")
    update_target_freq: int = field(default=10000, metadata="frequency of updating the target net")


# Registering RLConfig class to enable duck typing
cs = ConfigStore.instance()
cs.store(name="config", node=RLConfig)
log = logging.getLogger(__name__)

@hydra.main(config_name="config")
def main(config: RLConfig) -> None:
    """ Runs a training experiment based on the given hydra configuration """

    print(f"Launched! Experiment logs available at {os.getcwd()}.")
    # and it's still not reproducible...
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = gym.envs.make(config.env)
    env.seed(config.seed)
    log.info(f"Training on '{config.env}'")

    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    agent = DQNAgent(action_size=action_size,
                     hidden_sizes=[state_size, *config.hidden_sizes, action_size],
                     memory_capacity=config.memory_capacity,
                     epsilon=config.epsilon,
                     discount_factor=config.discount_factor,
                     optimizer=config.optimizer,
                     learning_rate=config.lr,
                     use_target_net=config.use_target_net,
                     update_target_freq=config.update_target_freq)
    log.info(agent.Q_model)

    batch_size = config.batch_size
    global_steps = 0
    average_steps = []
    rewards = []
    episode_reward = 0

    for episode in range(config.num_episodes):
        state = env.reset()

        steps = 0
        while True:
            agent.set_epsilon(
                linear_epsilon_anneal(global_steps, config.exploration_steps))

            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            episode_reward += reward * (config.discount_factor**steps)
            state = next_state
            steps += 1

            if done:
                average_steps.append(steps)
                rewards.append(episode_reward)

                if (episode + 1) % 20 == 0:
                # if episode % 1 == 0:
                    log.info(f"Episode: {episode + 1:5d}/{config.num_episodes:5d}\t\t "
                             f"#Steps: {np.mean(average_steps):7.1f}\t\t "
                             f"Reward: {np.sum(rewards):7.1f}\t\t "
                             f"Max-|Q|: {agent.get_max_q_val():7.1f}\t\t "
                             f"Epsilon: {agent.epsilon:.2f}")

                    episode_reward = 0
                    average_steps = []
                    rewards = []
                    agent.clear_max_q_val()
                break

            if len(agent.memory) > batch_size:
                agent.train(batch_size, sample_memory=config.sample_memory)
                agent.update_target_network(global_steps)
                global_steps += 1

if __name__ == "__main__":
    main()
