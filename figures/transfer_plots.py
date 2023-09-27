import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax import struct
import distrax
import gymnax
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
from evosax import OpenES, ParameterReshaper

import sys

sys.path.insert(0, '..')
from purejaxrl.wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from policy_distillation.distill_brax import (
    Transition,
    wrap_brax_env,
    BCAgentContinuous,
    init_env,
    init_params,
    make_train
)
import time
import argparse
import pickle as pkl
import os
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()


def get_data(env):
    root = "/private/home/alupu/explainable-policies/results/ICLR/brax/no_virtual_batch_norm"

    all_data = {}

    for D in [64]:
        for seed in [0]:
            for overfit in [True, False]:
                if overfit:
                    folder = f"{root}/overfit/{env}/D{D}/seed{seed}/"
                else:
                    folder = f"{root}/distill/{env}/D{D}/seed{seed}/"

                with open(f"{folder}/data_final.pkl", "rb") as f:
                    all_data[(env, overfit, seed)] = pkl.load(f)

    return all_data


def get_transfer_plot(env):

    all_data = get_data(env)

    fitness = {}
    for overfit in [True, False]:

        data = all_data[(env, overfit, 0)]
        config = data["config"]
        es_config = data["es_config"]

        # Init environment and dataset (params)
        env, env_params = init_env(config)
        _, param_reshaper = init_params(env, env_params, es_config)
        # Replace params by best params
        final_dataset = param_reshaper.reshape_single(data["state"].mean)

        rng = jax.random.PRNGKey(0)

        @partial(jax.jit, static_argnames=['width', "epochs"])
        def get_fitness(width, epochs, lr, rng):
            new_config = config.copy()
            new_config["NUM_ENVS"] = 16
            new_config["UPDATE_EPOCHS"] = epochs
            new_config["WIDTH"] = width
            new_config["LR"] = lr

            new_train_fn = make_train(new_config)

            def new_BC_train(rng_input, dataset, action_labels):
                out = new_train_fn(dataset, action_labels, rng_input)
                return out  # ["metrics"]['returned_episode_returns'].mean()

            new_BC_train = jax.jit(new_BC_train)

            rng, rng_new = jax.random.split(rng)
            out_new = new_BC_train(rng_new, final_dataset["states"], final_dataset["actions"])

            returns = out_new["metrics"]["returned_episode_returns"]  # dim=(popsize, rollout_news, num_steps, num_envs)
            dones = out_new["metrics"]["returned_episode"]  # same dim, True for last steps, False otherwise
            fitness = (returns * dones).sum(axis=(-1, -2)) / dones.sum(axis=(-1, -2))  # fitness, dim = (popsize)

            return fitness

        vmapped_get_fitness = jax.vmap(get_fitness, in_axes=(None, None, 0, 0))

        widths = [32, 64, 128, 256, 512, 1024, 2056]

        lrs = jnp.array([0.001 * 2 ** i for i in range(10)])
        update_epochs = [x for x in range(100, 520, 20)]

        for width in widths:
            for epochs in update_epochs:
                rng, rng_run = jax.random.split(rng)
                batch_rngs = jax.random.split(rng_run, len(lrs))

                combo_fitness = vmapped_get_fitness(width, epochs, lrs, batch_rngs)
                fitness[(overfit, width, epochs)] = combo_fitness

    data_tuples = []
    for (overfit, width, epochs), fitness_values in fitness.items():
        for value in fitness_values:
            data_tuples.append((overfit, width, epochs, value.item()))

    # Convert list of tuples to DataFrame
    df = pd.DataFrame(data_tuples, columns=["Overfit", "Width", "Epochs", "Fitness"])

    df.to_pickle(f'{env}.pkl')

    # Create a violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Width', y='Fitness', data=df, scale='width',
                   color=sns.color_palette()[0], hue="Overfit", split=True)
    plt.title('Violin Plot of Fitness Distributions by Width')
    plt.xlabel('Width')
    plt.ylabel('Fitness')

    # Save the plot to a file
    plt.savefig(f'{env}.pdf')

    # Show the plot
    plt.show()


