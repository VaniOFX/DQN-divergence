import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import yaml
from omegaconf import DictConfig
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import re

results_dir = Path('experiments')
possible_experiment_settings = ('vanilla', 'memory', 'target', 'memory+target')

def get_experiment_setting(config):
    result = 'vanilla'
    if config.sample_memory and config.use_target_net:
        result = 'memory+target'
    elif config.sample_memory:
        result = 'memory'
    elif config.use_target_net:
        result = 'target'
    return result

def load_experiment_results(discount_factor=0.99, read_log=False, num_episodes=700):
    """
    Extract info from all experiments.
    
    Filter runs with specified discount_factor.

    Returns a dictionary: result[environment][experiment_setting][config][seed] -> dataframe for 1 run
    """
    result = {}

    for experiment_dir in list(results_dir.glob('*')):
        # skip yaml and other bs files
        if not experiment_dir.is_dir():
            continue

        hydra_dir = experiment_dir / '.hydra'
        config_file = hydra_dir / 'config.yaml'

        with open(config_file) as f:
            config = dict(yaml.safe_load(f))
        seed = config.pop('seed')
        config = DictConfig(config)

        environment = config.env
        experiment_setting = get_experiment_setting(config)
        
        if config.discount_factor == discount_factor and config.num_episodes == num_episodes:
            if environment not in result:
                result[environment] = {}
            if experiment_setting not in result[environment]:
                result[environment][experiment_setting] = {}
            if config not in result[environment][experiment_setting]:
                result[environment][experiment_setting][config] = {}
            # if same experiment and same seed, only use one of them and skip the rest
            if seed not in result[environment][experiment_setting][config]:
                if not read_log:
                    # some experiments may still be running, so exp_records not yet written
                    try:
                        data = pd.read_csv(experiment_dir / 'exp_records.csv', index_col=0)
                    except FileNotFoundError:
                        continue
                else:
                    log_file = experiment_dir / 'dqn.log'
                    with open(log_file) as f:
                        data = f.read()
                result[environment][experiment_setting][config][seed] = data

    return result

def extract_final_reward(log):
    """Extract final reward from log file."""
    last_reward_line = log.split('\n')[-3]
    return float(re.search('Avg-Episode-Reward:\s+(\S+)', last_reward_line).group(1))

def extract_reward_max_q(log):
    """Extract final reward and max Q from log file."""
    log_lines = log.split('\n')[8:-3]
    get_reward = lambda line: float(re.search('Avg-Episode-Reward:\s+(\S+)', line).group(1))
    get_q = lambda line: float(re.search('Max-\|Q\|:\s+(\S+)', line).group(1))
    rewards = [get_reward(line) for line in log_lines]
    max_qs = [get_q(line) for line in log_lines]
    return np.array([rewards, max_qs])

def extract_final_max_q(results):
    """Extract (latest) max q from pandas dataframe."""
    return results.loc[results.index[-1], 'max_q_values']

def extract_q_values(results):
    """Extract all the max q values"""
    return results['max_q_values'].tolist()

def iterate_results(results, extract_fn):
    """
    Iterate experiment results and extract info using some extraction function.
    """
    outputs = {}
    for environment, environment_results in results.items():
        if environment not in outputs:
            outputs[environment] = {}
        for experimental_setting, setting_results in environment_results.items():
            outputs[environment][experimental_setting] = []
            for config, seeds_results in setting_results.items():
                for seed, actual_results in seeds_results.items():
                    output = extract_fn(actual_results)
                    outputs[environment][experimental_setting].append(output)
            outputs[environment][experimental_setting] = np.array(outputs[environment][experimental_setting])
    return outputs


def make_violinplots(data, discount_factor=None, environment=None, figsize=(10,10),
                     mode='q_divergence', save_path=None):
    """
    Make violinplots similar to van Hassalt et al.
    
    q_vals input from 'get_max_q_values' function.
    
    If environment is not specified, merge all environments together for the plot.
    """
    assert mode in ('q_divergence', 'reward')
    data_dfs = {}
    for env, vals in data.items():
        # deal with irregular experiment lengths
        data_dfs[env] = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vals.items()]))
        # max_q_dfs[env] = pd.DataFrame(vals)

    if environment is not None:
        df = data_dfs[environment]
    else:
        # merges all environments together
        df = pd.concat([df for df in data_dfs.values()])

    df_melt = pd.melt(df)
    # deal with irregular experiment lengths
    df_melt = df_melt[~df_melt.value.isna()]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df_melt, x='variable', y='value', order=possible_experiment_settings, scale='count',
                   cut=0, inner='stick', ax=ax, dpi=300)
    if mode == 'q_divergence':
        divergence_level = 1 / (1 - discount_factor)
        ax.axhline(divergence_level, linestyle='--', color='black', linewidth=3)
        ax.set_yscale('log')
        ylabel = 'Maximum |Q| Values (log scale)'
    elif mode == 'reward':
        ylabel = 'Returns'
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.grid(axis='y')
    ax.tick_params(axis='x', labelsize=15)
    if save_path is not None:
        plt.savefig(save_path)

def plot_q_values(data, discount_factor, figsize=(15,10), save_path=None):
    """
    Plot the maximum Q values over the episodes

    Aggregated across all seeds with error bands
    """
    #plt.style.use(["seaborn-talk","seaborn-deep"])
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    for exp_setting, q_values in data.items():
        mean_q_values = np.mean(q_values, axis=0)
        std_q_values = np.std(q_values, axis=0)
        upper_band = mean_q_values+std_q_values
        lower_band = mean_q_values-std_q_values
        episode_indexes = range(len(mean_q_values))
        ax.plot(episode_indexes, mean_q_values, label=exp_setting)
        ax.fill_between(episode_indexes, upper_band, lower_band, alpha=0.3)
        ax.set_yscale("log")
    ax.axhline(1 / (1 - discount_factor), linestyle='--', color='black', linewidth=3)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Maximum |Q| Values (log scale)")
    plt.legend()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()

def plot_rewards_qs(data, discount_factor, figsize=(7,7), save_path=None):
    """ Plot the maximum Q values against rewards for all tricks """
    # plt.style.use("seaborn-talk")
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    markers = ["o", "x", "s", "P"]
    # consistent ordering of exp_settings
    # data = OrderedDict(data)
    for i, exp_setting in enumerate(possible_experiment_settings):
        rewards_qs = data[exp_setting]
        rewards = rewards_qs[:, 0, -1]
        # mean_rewards = np.mean(rewards, axis=0)
        qs = rewards_qs[:, 1, -1]
        # mean_qs = np.mean(qs, axis=0)
        ax.scatter(qs, rewards, marker=markers[i], label=exp_setting, alpha=0.7)
    ax.axvline(1 / (1 - discount_factor), linestyle='--', color='black', linewidth=1)
    ax.set_xlabel("Maximum |Q| Values")
    ax.set_ylabel("Average Returns per Episode")
    plt.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    plt.style.use("fivethirtyeight")
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', metavar='discount_factor', type=float, default=0.99, help='discount_factor to base plotting on')
    parser.add_argument('--num_episodes', metavar='num_episodes', type=int, default=700, help='number of episodes')
    parser.add_argument('--environment', metavar='training environment', type=str, default='MountainCar-v0', help='which environment for plots')
    parser.add_argument('--reward', action='store_true', help='if specified, make plots for average reward')
    parser.add_argument('--q_values', action='store_true', help='plot maximum Q values across episodes')
    parser.add_argument('--reward_q_values', action='store_true', help='plot rewards vs q values')
    args = parser.parse_args()

    environment = None
    try:
        environment = args.environment
    except AttributeError:
        pass

    if args.q_values:
        results = load_experiment_results(discount_factor=args.discount, num_episodes=args.num_episodes)
        q_values_df = iterate_results(results, extract_fn=extract_q_values)
        plot_q_values(q_values_df[environment], discount_factor=args.discount, save_path=f"{environment}_q_values.png")
    elif args.reward_q_values:
        logs = load_experiment_results(read_log=True)
        rewards_qs = iterate_results(logs, extract_fn=extract_reward_max_q)
        plot_rewards_qs(rewards_qs[environment], discount_factor=args.discount, save_path=f"{environment}_rewards_q_values.png")
        pass
    else:
        mode = 'reward' if args.reward else 'q_divergence'
        if mode == 'reward':
            read_log = True
            extract_fn = extract_final_reward
        else:
            read_log = False
            extract_fn = extract_final_max_q

        results = load_experiment_results(discount_factor=args.discount, read_log=read_log, num_episodes=args.num_episodes)
        data = iterate_results(results, extract_fn=extract_fn)
        make_violinplots(data, discount_factor=args.discount, mode=mode, environment=environment,
                        save_path=f'violinplot_{mode}_{args.environment}_{args.discount}.png')