import gym
import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax

from random_env.envs import RandomEnvDiscreteActions as REDA

name = 'MountainCar-v0'
env = gym.make(name)
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")

n_obs = 2
n_act = 1

state_feature_size = 10
def func_pi(state_features):
    weights = hk.Linear(state_feature_size, w_init=jnp.zeros)

    mu = hk.Sequential((
        weights,
        hk.Linear(n_act, w_init=jnp.zeros)
    ))

pi = coax.P






