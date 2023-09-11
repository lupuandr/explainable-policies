import jax
import jax.numpy as jnp

from jax import grad, jit, vmap
from jax import random

import flax.linen as nn
# import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal, normal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
# import gymnax
# from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import time

import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from jax.scipy.special import logsumexp

from evosax import OpenES, ParameterReshaper, NetworkMapper

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "--net",
        type=str,
        help="Neural net type (mlp/cnn)",
        default="mlp"
    )
args = parser.parse_args()
args.net = args.net.lower()

# 1. LOAD DATA

batch_size = 128 # irrelevant
n_targets = 10 # number of classes

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
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
        return np.ravel(np.array(pic, dtype=np.float32))
    
class Cast(object):
    def __call__(self, pic):
        return np.array(pic, dtype=np.float32).reshape(1, 28, 28, 1)
    
# Define our dataset, using torch datasets
if args.net == "mlp":
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
else:
    mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=Cast())
# training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.data).reshape(len(mnist_dataset.data), 1, 28, 28, 1)
train_labels = jax.nn.one_hot(np.array(mnist_dataset.targets), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = np.array(mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), 1, 28, 28, 1), dtype=np.float32) #
test_labels = jax.nn.one_hot(np.array(mnist_dataset_test.targets), n_targets)

if args.net == "mlp":
    # Flatten observations
    train_images = train_images.reshape(len(mnist_dataset.data), -1)
    test_images = test_images.reshape(len(mnist_dataset_test.data), -1)
   
    
# def get_datasets():
#     """Load MNIST train and test datasets into memory."""
#     ds_builder = tfds.builder('mnist')
#     ds_builder.download_and_prepare()
#     train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
#     test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
#     train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
#     test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
#     return train_ds, test_ds

# train_ds, test_ds = get_datasets()

# breakpoint()
# train_images = train_ds["image"]
# train_labels = train_ds["label"]

# test_images = train_ds["image"]
# test_labels = train_ds["label"]

# print("Data loaded. Images of shape:")
# print(test_images[0].shape)


# 2. DEFINE NN

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
 
# WHY IS THIS IN RED???
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        pi = distrax.Categorical(logits=x)
        
        return pi

    
# 3. INITIALIZE NN, TRAIN_STATE
seed = 0
rng = jax.random.PRNGKey(seed)
rng, _rng = jax.random.split(rng)

if args.net == "mlp":
    learning_rate = 0.005
    num_epochs = 10
    # batch_size = 128
    # n_targets = 10
    momentum=0

    network = MLP(10, width=512, activation="relu")
    init_x = jnp.zeros(784)
    network_params = network.init(_rng, init_x)
    print("Initialized MLP")
    
elif args.net == "cnn":
    
    # CNN hparams
    learning_rate = 0.005
    num_epochs = 10
    # batch_size = 128
    # n_targets = 10
    momentum=0.9
    
    network = CNN()
    init_x = jnp.zeros((1, 28, 28, 1))
    network_params = network.init(_rng, init_x,)
    print("Initialized CNN")

config = {
    "LR" : learning_rate,
    "MOMENTUM" : momentum
}

tx = optax.sgd(config["LR"], momentum=config["MOMENTUM"])

train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

batched_predict = vmap(train_state.apply_fn, in_axes=(None, 0))

print(f"NN initialized, type {type(network)}")


# 4. DEFINE KEY METHODS
def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    pred_probs = batched_predict(params, images).probs
    # Reshaping accounts for CNN having an extra batched dimension, which otherwise messes broadcasting
    pred_probs = pred_probs.reshape(pred_probs.shape[0],-1)
    predicted_class = jnp.argmax(pred_probs, axis=1)
    return jnp.mean(predicted_class == target_class)

# if args.net == "MLP":
#     def loss(params, images, targets):
#         preds = batched_predict(params, images)
#         return -jnp.mean(preds.logits.reshape(targets.shape) * targets)
# elif args.net == "CNN":
#     def loss(params, images, targets):
#         preds = batched_predict(params, images)
#         return -jnp.mean(preds.logits.reshape(targets.shape) * targets)
def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds.logits.reshape(targets.shape) * targets)

@jit
def update(train_state, x, y):
    grad_fn = jax.value_and_grad(loss, has_aux=False)
    loss_val, grads = grad_fn(train_state.params, x, y)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss_val, -1, #jnp.linalg.norm(grads)
    
    
# 5. GET CLASS-WISE AVERAGE DATASET
x_list = []
y_list = []
imgs_per_class = 1
for digit in range(10):
    mean_digit = train_images[mnist_dataset.targets == digit].mean(0)
    x_list.append(mean_digit)
    y_list.append(digit)

x_list = jnp.array(x_list)
y_list = jnp.array(y_list)

print("Datasets shapes")
print(x_list.shape)
print(y_list.shape)


# 6. TRAIN
num_epochs = 8

# with jax.disable_jit(False):
#     for epoch in range(num_epochs):
#         start_time = time.time()
#         for x, y in training_generator:
#             y = jax.nn.one_hot(y, n_targets)
# #             breakpoint()
#             train_state, loss_val, grad_norm = update(train_state, x, y)
# #             breakpoint()
#         epoch_time = time.time() - start_time

#         train_acc = accuracy(train_state.params, train_images, train_labels)
#         test_acc = accuracy(train_state.params, test_images, test_labels)
#         print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
#         print("Training set accuracy {}".format(train_acc))
#         print("Test set accuracy {}".format(test_acc))
#         print("Train loss {}, grad norm {}".format(loss_val, grad_norm))
#         print()

for epoch in range(num_epochs):
    start_time = time.time()
    for i in range(100):
        y = jax.nn.one_hot(y_list, n_targets)
        train_state, loss_val, grad_norm = update(train_state, x_list, y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(train_state.params, train_images, train_labels)
    test_acc = accuracy(train_state.params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
    print("Train loss {}, grad norm {}".format(loss_val, grad_norm))
    print()