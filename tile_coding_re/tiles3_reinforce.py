import os
from copy import deepcopy
import numpy as np
import gym
import jax
import jax.numpy as jnp
import coax
import optax
import haiku as hk

from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions, REDAX

tensorboard_dir = os.path.join('reinforce')

n_obs, n_act = 2, 2
actions = get_discrete_actions(n_act, 3)

env = REDAX(n_obs, n_act)
eval_env = deepcopy(env)
env = coax.wrappers.TrainMonitor(env, tensorboard_dir=tensorboard_dir, tensorboard_write_all=True)

ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
tilings = Tilings(nb_tilings=8, nb_bins=2, feature_ranges=ranges, max_tiles=2**20)
qvf = QValueFunctionTiles3(tilings, actions)


def func_pi(S, is_training):
    seq = hk.Sequential((
        hk.Linear(8, w_init=jnp.zeros), jax.nn.relu,
        hk.Linear(8, w_init=jnp.zeros), jax.nn.relu,
        hk.Linear(8, w_init=jnp.zeros), jax.nn.relu,
        hk.Linear(len(actions), w_init=jnp.zeros), jax.nn.softmax
    ))
    return dict(logits=seq(S))


pi = coax.Policy(func_pi, env)


# specify how to update policy and value function
vanilla_pg = coax.policy_objectives.VanillaPG(pi, optimizer=optax.adam(1e-2))

tracer = coax.reward_tracing.MonteCarlo(gamma=0.9)

eval_every = 1000

for ep in range(1000):
    o = env.reset()
    for t in range(env.EPISODE_LENGTH_LIMIT):
        a, logp = pi(o, return_logp=True)
        otp1, r, d, _ = env.step(a)
        tracer.add(o, a, r,d, logp)

        if (env.T+1) % eval_every == 0:
            eplens = []
            for _ in range(20):
                o_eval = eval_env.reset()
                d_eval = False
                step = 0
                while not d_eval:
                    a_eval = pi(o_eval)
                    otp1_eval, r_eval, d_eval, _ = eval_env.step(a_eval)
                    o_eval = otp1_eval.copy()
                    step += 1
                eplens.append(step)
            env.record_metrics({'episode/eplens':np.mean(eplens)})

        while tracer:
            batch = tracer.pop()
            Gn = batch.Rn

            lr = 0.1 * 0.9**(ep//10)
            err = qvf.update(batch.S[0], actions[0], Gn[0], 0.1/8)
            env.record_metrics({'qvf/error':err})
            env.record_metrics({'qvf/lr':lr})

            Gn_pred = qvf.value(batch.S[0])

            Adv = Gn - Gn_pred
            env.record_metrics({'qvf/abs_Adv': abs(Adv)})
            metrics = vanilla_pg.update(batch, Adv=Adv)
            env.record_metrics(metrics)

        if d:
            break
