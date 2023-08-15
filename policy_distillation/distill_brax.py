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

import wandb

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
import time
import argparse
import pickle as pkl
import os


def wrap_brax_env(env, normalize=True, gamma=0.99):
    """Apply standard set of Brax wrappers"""
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if normalize:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma)
    return env

# Continuous action BC agent
class BCAgentContinuous(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    width: int = 64 #256 for Brax

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    """Create training function based on config. The returned function will:
    - Train a policy through BC
    - Evaluate the policy in the environment
    """
    config["NUM_UPDATES"] = config["UPDATE_EPOCHS"]

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    # Do I need a schedule on the LR for BC?
    def linear_schedule(count):
        frac = 1.0 - (count // config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(synth_data, action_labels, rng):
        """Train using BC on synthetic data with fixed action labels and evaluate on RL environment"""

        action_shape = env.action_space(env_params).shape[0]
        network = BCAgentContinuous(
            action_shape, activation=config["ACTIVATION"], width=config["WIDTH"]
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        assert (
                synth_data[0].shape == env.observation_space(env_params).shape
        ), f"Data of shape {synth_data[0].shape} does not match env observations of shape {env.observation_space(env_params).shape}"

        # Setup optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # Train state carries everything needed for NN training
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # 2. BC TRAIN LOOP
        def _bc_train(train_state, rng):
            def _bc_update_step(bc_state, unused):
                train_state, rng = bc_state

                def _loss_and_acc(params, apply_fn, step_data, y_true, num_classes, grad_rng):
                    """Compute cross-entropy loss and accuracy."""
                    pi = apply_fn(params, step_data)
                    y_pred = pi.sample(seed=grad_rng)

                    acc = jnp.mean(jnp.abs(y_pred - y_true))
                    log_prob = -pi.log_prob(y_true)
                    loss = jnp.sum(log_prob)
                    #                     loss = jnp.sum(jnp.abs(y_pred - y_true))
                    loss /= y_true.shape[0]

                    return loss, acc

                grad_fn = jax.value_and_grad(_loss_and_acc, has_aux=True)

                # Not needed if using entire dataset
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, len(action_labels))
                step_data = synth_data[perm]
                y_true = action_labels[perm]

                rng, grad_rng = jax.random.split(rng)

                loss_and_acc, grads = grad_fn(
                    train_state.params,
                    train_state.apply_fn,
                    step_data,
                    y_true,
                    action_shape,
                    grad_rng
                )
                train_state = train_state.apply_gradients(grads=grads)
                bc_state = (train_state, rng)
                return bc_state, loss_and_acc

            bc_state = (train_state, rng)
            bc_state, loss_and_acc = jax.lax.scan(
                _bc_update_step, bc_state, None, config["UPDATE_EPOCHS"]
            )
            loss, acc = loss_and_acc
            return bc_state, loss, acc

        rng, _rng = jax.random.split(rng)
        bc_state, bc_loss, bc_acc = _bc_train(train_state, _rng)
        train_state = bc_state[0]

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        #         obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv, env_state = env.reset(reset_rng, env_params)

        # 3. POLICY EVAL LOOP
        def _eval_ep(runner_state):
            # Environment stepper
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select Action
                rng, _rng = jax.random.split(rng)
                pi = train_state.apply_fn(train_state.params, last_obs)
                if config["GREEDY_ACT"]:
                    action = pi.argmax(
                        axis=-1
                    )  # if 2+ actions are equiprobable, returns first
                else:
                    action = pi.sample(seed=_rng)

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                #                 obsv, env_state, reward, done, info = jax.vmap(
                #                     env.step, in_axes=(0, 0, 0, None)
                #                 )(rng_step, env_state, action, env_params)
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, -1, reward, pi.log_prob(action), last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = _eval_ep(runner_state)

        metric["bc_loss"] = bc_loss
        metric["bc_accuracy"] = bc_acc

        metric["states"] = synth_data
        metric["action_labels"] = action_labels
        metric["rng"] = rng

        return {"runner_state": runner_state, "metrics": metric}

    return train


def init_env(config):
    """Initialize environment"""
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = wrap_brax_env(env, normalize=config["NORMALIZE_ENV"])
    return env, env_params


def init_params(env, env_params, es_config):
    """Initialize dataset to be learned"""
    params = {
        "states": jnp.zeros((es_config["dataset_size"], *env.observation_space(env_params).shape)),
        "actions": jnp.zeros((es_config["dataset_size"], *env.action_space(env_params).shape))
    }
    param_reshaper = ParameterReshaper(params)
    return params, param_reshaper


def init_es(rng_init, param_reshaper, es_config):
    """Initialize OpenES strategy"""
    strategy = OpenES(
        popsize=es_config["popsize"],
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        maximize=True,
    )
    # Replace state mean with real observations
    # state = state.replace(mean = sampled_data)

    es_params = strategy.default_params
    # es_params = es_params.replace(init_max=1.0)
    state = strategy.initialize(rng_init, es_params)

    return strategy, es_params, state


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Default arguments should result in ~1600 return in Hopper

    # Outer loop args
    parser.add_argument(
        "--env",
        type=str,
        help="Brax environment name",
        default="hopper"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Number of state-action pairs",
        default=4,
    )
    parser.add_argument(
        "--popsize",
        type=int,
        help="Number of state-action pairs",
        default=512
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="Number of ES generations",
        default=200
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        help="Number of BC policies trained per candidate",
        default=16
    )

    # Inner loop args
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of BC epochs in the inner loop",
        default=20
    )
    parser.add_argument(
        "--eval_envs",
        type=int,
        help="Number of evaluation environments",
        default=16
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="NN nonlinearlity type (relu/tanh)",
        default="tanh"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="NN width",
        default=64
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="NN learning rate",
        default=5e-3
    )

    # Misc. args
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
        default=1337
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Num. generations between logs",
        default=1
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to save folder",
        default="../results/"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    if args.folder[-1] != "/":
        args.folder = args.folder + "/"

    return args


if __name__ == "__main__":

    args = parse_arguments()

    config = {
        "LR": args.lr,  # 3e-4 for Brax?
        "NUM_ENVS": args.eval_envs,  # 8 # Num eval envs for each BC policy
        "NUM_STEPS": 1024,  # 128 # Max num eval steps per env
        "UPDATE_EPOCHS": args.epochs,  # Num BC gradient steps
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": args.activation,
        "WIDTH": args.width,
        "ENV_NAME": args.env,
        "ANNEAL_LR": False,  # False for Brax?
        "GREEDY_ACT": False,  # Whether to use greedy act in env or sample
        "ENV_PARAMS": {},
        "GAMMA": 0.99,
        "NORMALIZE_ENV": True,
        "DEBUG": args.debug,
    }
    es_config = {
        "popsize": args.popsize,  # Num of candidates (variations) generated every generation
        "dataset_size": args.dataset_size, # Num of (s,a) pairs
        "rollouts_per_candidate": args.rollouts,  # 32 Num of BC policies trained per candidate
        "n_generations": args.generations,
        "log_interval": args.log_interval,
    }

    print("config")
    print("-----------------------------")
    for k, v in config.items():
        print(f"{k} : {v},")
    print("-----------------------------")
    print("ES_CONFIG")
    for k, v in es_config.items():
        print(f"{k} : {v},")

    # Setup wandb
    wandb_config = config.copy()
    wandb_config["es_config"] = es_config
    wandb_run = wandb.init(project="Policy Distillation", config=wandb_config)
    wandb.define_metric("D")
    wandb.summary["D"] = es_config["dataset_size"]
#     wandb.define_metric("mean_fitness", summary="last")
#     wandb.define_metric("max_fitness", summary="last")
    

    # Init environment and dataset (params)
    env, env_params = init_env(config)
    params, param_reshaper = init_params(env, env_params, es_config)

    rng = jax.random.PRNGKey(args.seed)

    # Initialize OpenES Strategy
    rng, rng_init = jax.random.split(rng)
    strategy, es_params, state = init_es(rng_init, param_reshaper, es_config)

    # Set up vectorized fitness function
    train_fn = make_train(config)

    def single_seed_BC(rng_input, dataset, action_labels):
        out = train_fn(dataset, action_labels, rng_input)
        return out

    multi_seed_BC = jax.vmap(single_seed_BC, in_axes=(0, None, None))  # Vectorize over seeds
    train_and_eval = jax.jit(
        jax.vmap(multi_seed_BC, in_axes=(None, 0, 0)))  # Vectorize over datasets

    if len(jax.devices()) > 1:
        # If available, distribute over multiple GPUs
        train_and_eval = jax.pmap(train_and_eval, in_axes=(None, 0, 0))

    start = time.time()
    lap_start = start
    fitness_over_gen = []
    max_fitness_over_gen = []
    for gen in range(es_config["n_generations"]):
        # Gen new dataset
        rng, rng_ask, rng_inner = jax.random.split(rng, 3)
        datasets, state = jax.jit(strategy.ask)(rng_ask, state, es_params)
        # Eval fitness
        batch_rng = jax.random.split(rng_inner, es_config["rollouts_per_candidate"])
        # Preemptively overwrite to reduce memory load
        out = None
        returns = None
        dones = None
        fitness = None
        shaped_datasets = None

        with jax.disable_jit(False):
            shaped_datasets = param_reshaper.reshape(datasets)

            out = train_and_eval(batch_rng, shaped_datasets["states"], shaped_datasets["actions"])

            returns = out["metrics"]["returned_episode_returns"]  # dim=(popsize, rollouts, num_steps, num_envs)
            dones = out["metrics"]["returned_episode"]  # same dim, True for last steps, False otherwise

            # Division by zero, watch out
            fitness = (returns * dones).sum(axis=(-1, -2, -3)) / dones.sum(
                axis=(-1, -2, -3))  # fitness, dim = (popsize)
            fitness = fitness.flatten()  # Necessary if pmap-ing to 2+ devices
        #         fitness = jnp.minimum(fitness, fitness.mean()+40)

        # Update ES strategy with fitness info
        state = jax.jit(strategy.tell)(datasets, fitness, state, es_params)
        fitness_over_gen.append(fitness.mean())
        max_fitness_over_gen.append(fitness.max())

        # Logging
        if gen % es_config["log_interval"] == 0 or gen == 0:
            lap_end = time.time()
            if len(jax.devices()) > 1:
                bc_loss = out["metrics"]["bc_loss"][:, :, :, -1]
                bc_acc = out["metrics"]["bc_accuracy"][:, :, :, -1]
            else:
                bc_loss = out["metrics"]["bc_loss"][:, :, -1]
                bc_acc = out["metrics"]["bc_accuracy"][:, :, -1]

            print(
                f"Gen: {gen}, Fitness: {fitness.mean():.2f} +/- {fitness.std():.2f}, "
                + f"Best: {state.best_fitness:.2f}, BC loss: {bc_loss.mean():.2f} +/- {bc_loss.std():.2f}, "
                + f"BC mean error: {bc_acc.mean():.2f} +/- {bc_acc.std():.2f}, Lap time: {lap_end - lap_start:.1f}s"
            )
            wandb.log({
                f"{config['ENV_NAME']}:mean_fitness" : fitness.mean(),
                f"{config['ENV_NAME']}:fitness_std" : fitness.std(),
                f"{config['ENV_NAME']}:max_fitness" : fitness.max(),
                "mean_fitness" : fitness.mean(),
                "max_fitness" : fitness.max(),
                "BC_loss" : bc_loss.mean(),
                "BC_accuracy" : bc_acc.mean(),
                "Gen time" : lap_end - lap_start,
            })
            lap_start = lap_end
    print(f"Total time: {(lap_end - start) / 60:.1f}min")

    data = {
        "state": state,
        "fitness_over_gen": fitness_over_gen,
        "max_fitness_over_gen": max_fitness_over_gen,
        "fitness": fitness,
        "config": config,
        "es_config": es_config
    }


    # TODO: Change path
    directory = args.folder + f"brax_{config['ENV_NAME']}/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    filename = directory + f"D{es_config['dataset_size']}/"
    filename = filename + f"{config['ACTIVATION']}E{config['UPDATE_EPOCHS']}P{es_config['popsize']}{config['WIDTH']}.pkl"
    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()