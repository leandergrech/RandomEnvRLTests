import coax
import gym
import haiku as hk
import jax
import jax.numpy as jnp
from coax.value_losses import mse
from optax import adam

name = 'dqn'

env = gym.make('CartPole-v0')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func(S, is_training):
    """ type-2 q-function: s -> q(s,.) """
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env.action_space.n, w_init=jnp.zeros)
    ))
    return seq(S)


# value function and its derived policy
q = coax.Q(func, env)
pi = coax.BoltzmannPolicy(q, temperature=0.1)

# target network
q_targ = q.copy()

# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=100000)

# updater
qlearning = coax.td_learning.QLearning(q, q_targ=q_targ, loss_function=mse, optimizer=adam(0.001))


# for I in range(5):
#     coax.utils.render_episode(env, policy=pi.mode)

# train
for ep in range(500):
    s = env.reset()
    for t in range(env.spec.max_episode_steps):
        a = pi(s)
        stp1, r, d, _ = env.step(a)

        # extend last reward as asymptotic best-case return
        if t == env.spec.max_episode_steps - 1:
            assert d
            r = 1 / (1 - tracer.gamma)

        # trace rewards and add transition to replay buffer
        tracer.add(s, a, r, d)
        while tracer:
            buffer.add(tracer.pop())

        # learn
        if len(buffer) >= 100:
            transition_batch = buffer.sample(batch_size=32)
            metrics = qlearning.update(transition_batch)
            env.record_metrics(metrics)

        # sync target network
        q_targ.soft_update(q, tau=0.01)

        if d:
            break

        s = stp1

    # early stopping
    if env.avg_G > env.spec.reward_threshold:
        break

# run env on more time to render
for I in range(10):
    coax.utils.render_episode(env, policy=pi.mode)#, filepath=f"./data/{name}.gif", duration=25)
