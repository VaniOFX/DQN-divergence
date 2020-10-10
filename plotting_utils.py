import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import yaml
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse

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

def load_experiment_results(discount_factor=0.99):
    """
    Extract info from all experiments.
    
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
        
        if config.discount_factor == discount_factor:
            if environment not in result:
                result[environment] = {}
            if experiment_setting not in result[environment]:
                result[environment][experiment_setting] = {}
            if config not in result[environment][experiment_setting]:
                result[environment][experiment_setting][config] = {}
            # if same experiment and same seed, only use one of them and skip the rest
            if seed not in result[environment][experiment_setting][config]:
                # some experiments may still be running, so exp_records not yet written
                try:
                    df = pd.read_csv(experiment_dir / 'exp_records.csv', index_col=0)
                except FileNotFoundError:
                    continue
                result[environment][experiment_setting][config][seed] = df
    return result


def get_max_q_values(results):
    """
    Extract max q values from experiment results, which is output from `load_experiment_results`.
    
    Returns a dictionary: result[environment][experimential_setting] -> np array max q value for each run
    """
    max_q_values = {}
    for environment, environment_results in results.items():
        if environment not in max_q_values:
            max_q_values[environment] = {}
        for experimental_setting, setting_results in environment_results.items():
            max_q_values[environment][experimental_setting] = []
            for config, seeds_results in setting_results.items():
                for seed, actual_results in seeds_results.items():
                    max_q_value = actual_results.loc[actual_results.index[-1], 'max_q_values']
                    max_q_values[environment][experimental_setting].append(max_q_value)
            max_q_values[environment][experimental_setting] = np.array(max_q_values[environment][experimental_setting])
            
    return max_q_values


def make_violinplots(q_vals, discount_factor, environment=None, figsize=(15,10), save_path=None):
    """
    Make violinplots similar to van Hassalt et al.
    
    q_vals input from 'get_max_q_values' function.
    
    If environment is not specified, merge all environments together for the plot.
    """
    max_q_dfs = {}
    for env, vals in max_q_values.items():
        # deal with irregular experiment lengths
        max_q_dfs[env] = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vals.items()]))
        # max_q_dfs[env] = pd.DataFrame(vals)

    if environment is not None:
        df = max_q_dfs[environment]
    else:
        # merges all environments together
        df = pd.concat([df for df in max_q_dfs.values()])

    df_melt = pd.melt(df)
    # deal with irregular experiment lengths
    df_melt = df_melt[~df_melt.value.isna()]

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df_melt, x='variable', y='value', order=possible_experiment_settings, scale='count',
                   cut=0, inner='stick', ax=ax)
    divergence_level = 1 / (1 - discount_factor)
    ax.axhline(divergence_level, linestyle='--', color='black', linewidth=3)
    ax.set_xlabel(None)
    ax.set_ylabel('max abs Q', fontsize=20)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.grid(axis='y')
    ax.tick_params(axis='x', labelsize=15)
    if save_path is not None:
        plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', metavar='discount_factor', type=float, default=0.6, help='discount_factor to base plotting on')
    parser.add_argument('--environment', metavar='training environment', type=str, default='cartpole', help='which environment for plots')
    args = parser.parse_args()

    environment = None
    try:
        environment = args.environment
    except AttributeError:
        pass

    results = load_experiment_results(discount_factor=args.discount)
    max_q_values = get_max_q_values(results)
    make_violinplots(max_q_values, discount_factor=args.discount, environment=environment, save_path='violinplot.png')