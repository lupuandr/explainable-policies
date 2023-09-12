import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal, normal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax import struct
import distrax

from evosax import OpenES, ParameterReshaper
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST

import wandb

import sys
sys.path.insert(0, '..')
import time
import argparse
import pickle as pkl
import os

# 0.1 Model Defitions
class MLP(nn.Module):
    """Network architecture. Matches MinAtar PPO agent from PureJaxRL"""

    action_dim: Sequence[int]
    activation: str = "relu"
    width: int = 512

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.width, kernel_init=normal(), bias_init=normal()
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.width, kernel_init=normal(), bias_init=normal()
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=normal(), bias_init=normal()
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class CNN(nn.Module):
    """A simple CNN model."""

    action_dim: Sequence[int]
    activation: str = "relu"
    ffwd_width: int = 256

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = activation(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.ffwd_width)(x)
        x = activation(x)
        x = nn.Dense(features=self.action_dim)(x)
        pi = distrax.Categorical(logits=x)

        return pi


# 0.2 Data loading helper methods
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        """Flatten and cast from Pic to Array for MLP use"""
        return jnp.ravel(jnp.array(pic, dtype=jnp.float32))


class Cast(object):
    def __call__(self, pic):
        """Cast from Pic to Array for CNN use"""
        # TODO: Make this more general, to work for CIFAR/ImageNet as well
        return jnp.array(pic, dtype=jnp.float32).reshape(1, 28, 28, 1)


def get_data(config):
    # TODO: Make this work for CIFAR as well
    datasets_dict = {"MNIST" : (MNIST, len(MNIST.classes)), "FashionMNIST" : (FashionMNIST, len(MNIST.classes))}
    DATASET, n_targets = datasets_dict[config["DATASET"]]
    dataset_name = config["DATASET"].lower()

    if config["NET"].lower() == "mlp":
        dataset = DATASET(f'/tmp/{dataset_name}/', download=True, transform=FlattenAndCast())
    else:
        dataset = DATASET(f'/tmp/{dataset_name}/', download=True, transform=Cast())

    # Get the full train dataset (for checking accuracy while training)
    # TODO: Make shape more general, i.e. for CIFAR
    train_images = np.array(dataset.data).reshape(len(dataset.data), 1, 28, 28, 1)
    train_labels = jax.nn.one_hot(np.array(dataset.targets), n_targets)

    # Get full test dataset
    dataset_test = DATASET('/tmp/mnist/', download=True, train=False)
    test_images = np.array(dataset_test.data.numpy().reshape(len(dataset_test.data), 1, 28, 28, 1),
                           dtype=np.float32)
    test_labels = jax.nn.one_hot(np.array(dataset_test.targets), n_targets)

    if config["NET"].lower() == "mlp":
        # Flatten observations
        train_images = train_images.reshape(len(dataset.data), -1)
        test_images = test_images.reshape(len(dataset_test.data), -1)

    if config["NORMALIZE"]:
        train_images = train_images/255
        test_images = test_images/255

    return train_images, train_labels, test_images, test_labels, n_targets


def make_train(config, train_images, train_labels, n_targets):
    """Create training function based on config. The returned function will:
    - Train a classifier on synthetic data
    - Evaluate the classifier on the real train data
    """
    config["NUM_UPDATES"] = config["UPDATE_EPOCHS"]

    # Do I need a schedule on the LR for BC?
    def linear_schedule(count):
        frac = 1.0 - (count // config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(synth_data, synth_targets, rng):
        """Train using BC on synthetic data with fixed action labels and evaluate on RL environment"""
        if config["NET"].lower() == "mlp":
            network = MLP(n_targets, activation=config["ACTIVATION"], width=config["WIDTH"])
        elif config["NET"].lower() == "cnn":
            network = CNN(n_targets, activation=config["ACTIVATION"], ffwd_width=config["WIDTH"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(train_images[0].shape)
        network_params = network.init(_rng, init_x)

        assert (
                synth_data[0].shape == train_images[0].shape
        ), f"Synth data of shape {synth_data[0].shape} does not match real data of shape {train_images[0].shape}"

        # Setup optimizer
        if config["ANNEAL_LR"]:
            tx = optax.sgd(learning_rate=linear_schedule, momentum=config["MOMENTUM"])
        else:
            tx = optax.sgd(config["LR"], momentum=config["MOMENTUM"])

        # Train state carries everything needed for NN training
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        batched_predict = jax.vmap(train_state.apply_fn, in_axes=(None, 0))

        # LOSS AND ACCURACY
        def _loss_and_acc(params, images, targets):
            """Compute cross-entropy loss and accuracy."""

            preds = batched_predict(params, images)
            loss = -jnp.mean(preds.logits.reshape(targets.shape) * targets)

            # Reshaping accounts for CNN having an extra batched dimension, which otherwise messes broadcasting
            pred_probs = preds.probs.reshape(preds.probs.shape[0], -1)
            acc = jnp.mean(jnp.argmax(pred_probs, axis=1) == jnp.argmax(targets, axis=1))

            return loss, acc

        # 2. BC TRAIN LOOP (kept BC naming, although it's just supervised here)
        def _bc_train(train_state, rng):
            def _bc_update_step(bc_state, unused):
                train_state, rng = bc_state

                grad_fn = jax.value_and_grad(_loss_and_acc, has_aux=True)

                # Not needed if using entire dataset
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, len(synth_targets))

                step_data = synth_data#[perm]
                y_true = synth_targets#[perm]

                rng, state_noise_rng, act_noise_rng = jax.random.split(rng, 3)
                state_noise = jax.random.normal(state_noise_rng, step_data.shape)
                act_noise = jax.random.normal(act_noise_rng, y_true.shape)

                step_data = step_data + config["DATA_NOISE"] * state_noise
                y_true = y_true + config["DATA_NOISE"] * act_noise

                rng, grad_rng = jax.random.split(rng)

                loss_and_acc, grads = grad_fn(
                    train_state.params,
                    step_data,
                    y_true,
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

        # 3. REAL DATA EVAL
        train_loss, train_acc = _loss_and_acc(train_state.params, train_images, train_labels)

        metric = {
            "synth_loss" : bc_loss,
            "synth_accuracy" : bc_acc,
            "train_loss" : train_loss,
            "train_accuracy" : train_acc,
            "synth_images" : synth_data,
            "synth_targets" : synth_targets,
            "rng" : rng
        }
        return {"fitness": train_loss, "metrics": metric, "train_state": train_state}

    return train


def init_params(rng, train_images, train_targets, es_config, n_targets):
    """Initialize dataset to be learned"""

    samples_per_class = es_config["dataset_size"] // n_targets
    if es_config["init_mode"] == "zero":
        images = jnp.zeros((es_config["dataset_size"], *train_images[0].shape))
        targets = jnp.zeros((es_config["dataset_size"], *train_targets[0].shape))

    elif es_config["init_mode"] == "mean":
        x_list = []
        y_list = []
        for target in range(n_targets):
            target_idxs = (train_targets == jax.nn.one_hot(target, n_targets)).all(1)
            mean_image = train_images[target_idxs].mean(0)
            x_list.extend([mean_image]*samples_per_class)
            y_list.extend([target]*samples_per_class)
        images = jnp.array(x_list)
        targets = jax.nn.one_hot(jnp.array(y_list), n_targets)

    elif es_config["init_mode"] == "sample":
        x_list = []
        y_list = []
        for target in range(n_targets):
            target_idxs = (train_targets == jax.nn.one_hot(target, n_targets)).all(1)
            target_images = train_images[target_idxs]
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, len(target_images))
            mean_images = [x for x in target_images[perm][0:samples_per_class]]
            x_list.extend(mean_images)
            y_list.extend([target]*samples_per_class)
        images = jnp.array(x_list)
        targets = jax.nn.one_hot(jnp.array(y_list), n_targets)

    if es_config["learn_labels"]:
        params = {"images": images, "targets": targets}
        fixed_targets = None
    else:
        params = {"images": images}
        # Fix targets to one-hots since we're not learning them
        y_list = []
        for target in range(n_targets):
            y_list.extend([target] * samples_per_class)
        fixed_targets = jax.nn.one_hot(jnp.array(y_list), n_targets)
    param_reshaper = ParameterReshaper(params)
    return params, param_reshaper, fixed_targets


def init_es(rng_init, param_reshaper, params, es_config):
    """Initialize OpenES strategy"""
    strategy = OpenES(
        popsize=es_config["popsize"],
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        maximize=False,         # Maximize=False because the fitness is the train loss
        lrate_init=es_config["lrate_init"]  # Passing it here since for some reason cannot update it in params.replace
    )
    # Replace state mean with real observations
    # state = state.replace(mean = sampled_data)

    es_params = strategy.params_strategy
    es_params = es_params.replace(sigma_init=es_config["sigma_init"], sigma_decay=es_config["sigma_decay"])
    state = strategy.initialize(rng_init, es_params)

    # Warm start with custom params (determined init_params)
    state = state.replace(mean=param_reshaper.flatten_single(params))

    return strategy, es_params, state


def parse_arguments(argstring=None):
    """Parse arguments either from `argstring` if not None or from command line otherwise"""
    parser = argparse.ArgumentParser()
    # Default arguments should result in ~1600 return in Hopper

    # Outer loop args
    parser.add_argument(
        "--dataset",
        type=str,
        help="MNIST/FashionMNIST",
        default="MNIST"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Number of state-action pairs",
        default=10,
    )
    parser.add_argument(
        "--popsize",
        type=int,
        help="ES population size",
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
        default=4
    )
    parser.add_argument(
        "--sigma_init",
        type=float,
        help="Initial ES variance",
        default=0.03
    )
    parser.add_argument(
        "--sigma_decay",
        type=float,
        help="ES variance decay factor",
        default=1.0
    )
    parser.add_argument(
        "--lrate_init",
        type=float,
        help="ES initial lrate",
        default=0.05
    )
    parser.add_argument(
        "--learn_labels",
        action="store_true",
        help="Whether to evolve labels (if False, fix labels to one-hots)",
        default=False
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        help="zero/sample/mean",
        default="zero"
    )

    # Inner loop args
    parser.add_argument(
        "--net",
        type=str,
        help="MLP / CNN",
        default="mlp"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of BC epochs in the inner loop",
        default=10
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="NN nonlinearlity type (relu/tanh)",
        default="relu"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="NN width",
        default=512
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="NN learning rate",
        default=5e-3
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="NN optimizer momentum",
        default=0
    )
    parser.add_argument(
        "--data_noise",
        type=float,
        help="Noise added to data during BC",
        default=0.0
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False
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
        "--log_dataset",
        action="store_true",
        help="Whether to log dataset to wandb",
        default=False
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
    if argstring is not None:
        args = parser.parse_args(argstring.split())
    else:
        args = parser.parse_args()

    if args.folder[-1] != "/":
        args.folder = args.folder + "/"

    return args


def make_configs(args):
    config = {
        "NET": args.net,
        "LR": args.lr,
        "MOMENTUM": args.momentum,
        "UPDATE_EPOCHS": args.epochs,  # Num supervised gradient steps
        "ACTIVATION": args.activation,
        "WIDTH": args.width,
        "DATASET": args.dataset,
        "ANNEAL_LR": False,  # False for Brax?
        "DATA_NOISE": args.data_noise,  # Add noise to data during BC training
        "NORMALIZE": args.normalize,
        "DEBUG": args.debug,
        "SEED": args.seed,
        "FOLDER": args.folder,
        "DATASET": args.dataset,
    }
    es_config = {
        "popsize": args.popsize,  # Num of candidates (variations) generated every generation
        "dataset_size": args.dataset_size,  # Num of (s,a) pairs
        "rollouts_per_candidate": args.rollouts,  # 32 Num of BC policies trained per candidate
        "n_generations": args.generations,
        "log_interval": args.log_interval,
        "sigma_init": args.sigma_init,
        "sigma_decay": args.sigma_decay,
        "lrate_init": args.lrate_init,
        "learn_labels": args.learn_labels,
        "init_mode": args.init_mode,
        "log_dataset": args.log_dataset
    }
    return config, es_config


def main(config, es_config):
    print("config")
    print("-----------------------------")
    for k, v in config.items():
        print(f"{k} : {v},")
    print("-----------------------------")
    print("ES_CONFIG")
    for k, v in es_config.items():
        print(f"{k} : {v},")
    print("-----------------------------")

    # Setup wandb
    if not config["DEBUG"]:
        wandb_config = config.copy()
        wandb_config["es_config"] = es_config

        wandb_run = wandb.init(project="Dataset Distillation", config=wandb_config)
        wandb.define_metric("D")
        wandb.summary["D"] = es_config["dataset_size"]

    # Get real datasets, with some preprocessing
    train_images, train_targets, test_images, test_targets, n_targets = get_data(config)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_init_params = jax.random.split(rng)

    # Initialize synthetic dataset
    params, param_reshaper, fixed_targets = init_params(rng_init_params, train_images, train_targets, es_config, n_targets)
    # Fixed targets, used if not targets are not learned
    samples_per_class = es_config["dataset_size"]//n_targets


    # Initialize OpenES Strategy
    rng, rng_init = jax.random.split(rng)
    strategy, es_params, state = init_es(rng_init, param_reshaper, params, es_config)

    # Set up vectorized fitness function
    train_fn = make_train(config, train_images, train_targets, n_targets)

    def single_seed_BC(rng_input, dataset, action_labels):
        out = train_fn(dataset, action_labels, rng_input)
        return out

    multi_seed_BC = jax.vmap(single_seed_BC, in_axes=(0, None, None))  # Vectorize over seeds
    if es_config["learn_labels"]:
        # vmap over images and labels
        train_and_eval = jax.jit(jax.vmap(multi_seed_BC, in_axes=(None, 0, 0)))  # Vectorize over datasets
    else:
        # vmap over images only
        train_and_eval = jax.jit(jax.vmap(multi_seed_BC, in_axes=(None, 0, None)))  # Vectorize over datasets

    if len(jax.devices()) > 1:
        # If available, distribute over multiple GPUs
        if es_config["learn_labels"]:
            # vmap over images and labels
            train_and_eval = jax.pmap(train_and_eval, in_axes=(None, 0, 0))
        else:
            # vmap over images only
            train_and_eval = jax.pmap(train_and_eval, in_axes=(None, 0, None))


    start = time.time()
    lap_start = start
    fitness_over_gen = []
    min_fitness_over_gen = []
    train_acc_over_gen = []
    for gen in range(es_config["n_generations"]):
        # Gen new dataset

        rng, rng_ask, rng_inner = jax.random.split(rng, 3)
        datasets, state = jax.jit(strategy.ask)(rng_ask, state, es_params)
        # Eval fitness
        batch_rng = jax.random.split(rng_inner, es_config["rollouts_per_candidate"])
        # Preemptively overwrite to reduce memory load
        out = None
        fitness = None
        shaped_datasets = None

        with jax.disable_jit(config["DEBUG"]):
            shaped_datasets = param_reshaper.reshape(datasets)
            if es_config["learn_labels"]:
                out = train_and_eval(batch_rng, shaped_datasets["images"], shaped_datasets["targets"])
            else:
                out = train_and_eval(batch_rng, shaped_datasets["images"], fixed_targets)

            # TODO: VERIFY FITNESS + accuracy IS COMPUTED ADEQUATELY, esp. mean
            # Loss on real train data
            fitness = out["metrics"]["train_loss"].mean(-1)     # mean over the different rollouts
            fitness = fitness.flatten()     # Necessary if pmap-ing to 2+ devices

        # Update ES strategy with fitness info
        state = jax.jit(strategy.tell)(datasets, fitness, state, es_params)
        fitness_over_gen.append(fitness.mean())
        min_fitness_over_gen.append(fitness.min())

        # Logging
        # TODO: ADD TEST RETURN
        if gen % es_config["log_interval"] == 0 or gen == 0:
            lap_end = time.time()
            if len(jax.devices()) > 1:
                # Take final loss and acc on synth dataset
                bc_loss = out["metrics"]["synth_loss"][:, :, :, -1]
                bc_acc = out["metrics"]["synth_accuracy"][:, :, :, -1]
            else:
                bc_loss = out["metrics"]["synth_loss"][:, :, -1]
                bc_acc = out["metrics"]["synth_accuracy"][:, :, -1]
            train_acc = out["metrics"]["train_accuracy"]

            print(
                f"Gen: {gen}, Fitness: {fitness.mean():.2f} +/- {fitness.std():.2f}, "
                + f"Best: {state.best_fitness:.2f}, BC loss: {bc_loss.mean():.2f} +/- {bc_loss.std():.2f}, "
                + f"Synth accuracy: {bc_acc.mean():.2f} +/- {bc_acc.std():.2f}, "
                + f"Train accuracy: {train_acc.mean():.2f} +/- {train_acc.std():.2f}, Lap time: {lap_end - lap_start:.1f}s"
            )
            if not config["DEBUG"]:
                log_dict = {
                    f"{config['DATASET']}:mean_fitness": fitness.mean(),
                    f"{config['DATASET']}:fitness_std": fitness.std(),
                    f"{config['DATASET']}:min_fitness": fitness.min(),
                    f"{config['DATASET']}:synth_accuracy": bc_acc.mean(),
                    f"{config['DATASET']}:train_accuracy": train_acc.mean(),
                    f"{config['DATASET']}:synth_loss": bc_loss.mean(),
#                     "mean_fitness": fitness.mean(),
#                     "min_fitness": fitness.min(),
#                     "synth_loss": bc_loss.mean(),
#                     "synth_accuracy": bc_acc.mean(),
#                     "train_accuracy": train_acc.mean(),
                    "Gen time": lap_end - lap_start,
                }


                if es_config["log_dataset"] and (gen % (es_config["log_interval"]*10) == 0 
                                                 or gen == 0 
                                                 or gen == es_config["log_interval"]-1):
                    final_dataset = param_reshaper.reshape_single(state.mean)
                    images = wandb.Image(
                        np.hstack(final_dataset["images"].reshape(-1, 28, 28, 1)),
                        caption="Final images"
                    )
                    final_labels = final_dataset["targets"] if es_config["learn_labels"] else fixed_targets
                    labels = wandb.Image(
                        np.array(final_labels),
                        caption="Final labels"
                    )
                    log_dict["Synth images"] = images
                    log_dict["Synth labels"] = labels

                wandb.log(log_dict)

            lap_start = lap_end
    print(f"Total time: {(lap_end - start) / 60:.1f}min")


    data = {
        "state": state,
        "fitness_over_gen": fitness_over_gen,
        "min_fitness_over_gen": min_fitness_over_gen,
        "fitness": fitness,
        "config": config,
        "es_config": es_config
    }

    directory = config["FOLDER"]
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + "data.pkl"
    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()


def train_from_arg_string(argstring):
    """Launches training from an argument string of the form
    `--dataset MNIST --popsize 1024 --epochs 200 ...`
    Main use case is in conjunction with Submitit for creating job arrays
    """
    args = parse_arguments(argstring)
    config, es_config = make_configs(args)
    main(config, es_config)


if __name__ == "__main__":
    args = parse_arguments()
    config, es_config = make_configs(args)
    main(config, es_config)
