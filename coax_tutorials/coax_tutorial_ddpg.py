import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from numpy import prod
import optax

name = 'ddpg-pendulum'

env = gym.make('Pendulum-v1')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f'./data/tensorboard/{name}')

def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
        hk.Reshape(env.action_space.shape)
    ))
    mu = seq(S)
    return dict(mu=mu, logvar=jnp.full_like(mu, -10)) # almost deterministic

def func_q(S, A, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    x = jnp.concatenate((S,A), axis=-1)
    return seq(x)

# main approximators
pi = coax.Policy(func_pi, env)
q = coax.Q(func_q, env, action_preprocessor=pi.proba_dist.preprocess_variate)

# target networks
q_targ = q.copy()
pi_targ = pi.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=25000)

# updaters
qlearning = coax.td_learning.QLearning(
    q, pi_targ, q_targ, loss_function=coax.value_losses.mse,
    optimizer=optax.adam(1e-3))
determ_pg = coax.policy_objectives.DeterministicPG(
    pi, q_targ, optimizer=optax.adam(1e-4))

# action noise
noise = coax.utils.OrnsteinUhlenbeckNoise(mu=0., sigma=.2, theta=.15) # oun bc env has element of momentum - more efficient exploration according to ddpg authors

# train
NB_INIT_SAMPLES = 5000
BATCH_SIZE = 128
TAU = 0.001
while env.T < int(1e6):
    s = env.reset()
    noise.reset()
    noise.sigma *= 0.99 # decrease slowly

    for t in range(env.spec.max_episode_steps):
        a = noise(pi(s)) # sample from main network
        stp1, r, d, _ = env.step(a)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, d)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= NB_INIT_SAMPLES: # collect some initial samples
            transition_batch = buffer.sample(batch_size=BATCH_SIZE)

            metrics = {'OUN/sigma':noise.sigma}
            metrics.update(determ_pg.update(transition_batch))
            metrics.update(qlearning.update(transition_batch))
            env.record_metrics(metrics)

            # sync target networks
            q_targ.soft_update(q, tau=TAU)
            pi_targ.soft_update(pi, tau=TAU)

        if d:
            break
        s = stp1

    if env.period(name='generate_gif', T_period=10000) and env.T > NB_INIT_SAMPLES:
        T = env.T - env.T % 10000
        coax.utils.generate_gif(
            env=env, policy=pi, filepath=f'./data/gifs/{name}/T{T:08d}.gif')



















