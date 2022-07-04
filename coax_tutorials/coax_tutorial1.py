import gym
import jax  # for arrays
import jax.numpy as jnp
import haiku as hk  # for nn
import optax  # for optimization
import coax

env = gym.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)


def forward_pass(S, is_training):
	lin = hk.Linear(env.action_space.n, w_init=jnp.zeros)
	return lin(S)


q = coax.Q(forward_pass, env)
pi = coax.BoltzmannPolicy(q, temperature=0.1)

# update
qlearning = coax.td_learning.QLearning(q)

# tracer
nstep = coax.reward_tracing.NStep(n=1, gamma=0.9)

# rendering
render = lambda: print(env.render(mode='ansi'))

for ep in range(500):
	s = env.reset()
	for t in range(env.spec.max_episode_steps):
		a = env.action_space.sample()
		stp1, r, d, info = env.step(a)

		# update
		nstep.add(s, a, r, d)
		while nstep:
			transition = nstep.pop()
			qlearning.update(transition)

		if d:
			break

		s = stp1

for _ in range(10):
	coax.render_episode(env, policy=pi.mode)
