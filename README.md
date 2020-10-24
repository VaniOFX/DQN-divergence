# Divergence in Deep Q-Learning: Two Tricks Are Better Than One

With this project we aim to analyse the effects of DQN's [1] tricks on divergence.
That is, we take a close look into Experience Replay and employing a separate Target Network.
What happens without these tricks? Can we get the same results with only one of them?
Are there any drawbacks to having both enabled?

All these can be answered with our DQN testbed.

## Setup

We have listed all required dependencies in the Conda `environment.yml` file.


## Obtaining results

We use Hydra for configuration.
A list of the default parameters can be obtained by running
```
$ python ./dqn.py --help
dqn is powered by Hydra.
[...]
== Config ==
Override anything in the config (foo.bar=value)

seed: 42
env: CartPole-v1
batch_size: 32
sample_memory: true
memory_capacity: 1000000
hidden_sizes:
- 128
num_episodes: 200
epsilon: 1.0
epsilon_lower_bound: 0.1
discount_factor: 0.99
exploration_steps: 400
optimizer: Adam
lr: 0.001
use_target_net: true
update_target_freq: 10000
log_freq: 20
```

These can easily be changed via the command line, and a search space of different parameters can also be explored:
```
$ python ./dqn.py discount_factor=0.99 use_target_net=False,True update_target_freq=2000 sample_memory=False,True num_episodes=700 env="Acrobot-v1" seed=24,23,22,21,20,19,18,17
```

Each experiment will result in a new subdirectory to be created under `./experiments/`.


## Visualising results

We provide a utility script to geenrate 4 different kinds of plots.
Namely, one can
* visualise the amount of soft divergence ($$\max|Q|$$) over time, throughout the episodes;
```
$ python ./plotting_utils.py --discount=0.99 --q_values --environment="Acrobot-v1"
```

* visualise a violin plot with the distribution of the $$max|Q|$$ (from the last batch):
```
$ python ./plotting_utils.py --discount=0.99 --environment="Acrobot-v1"
```

* visualise a violin plot with the distribution of the discounted returns (from the last batch):
```
$ python ./plotting_utils.py --discount=0.99 --rewards --environment="Acrobot-v1"
```

* and, most importantly, generate a scatter plot, outlining the connection between divergence and a good policy:
```
$ python ./plotting_utils.py --discount=0.99 --reward_q_values --environment="Acrobot-v1"
```

We have provided the results of our experiments under the `./img/` folder.


## References

[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.
