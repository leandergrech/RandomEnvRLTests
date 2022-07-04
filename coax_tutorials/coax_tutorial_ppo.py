import gym
import jax
import jax.numpy as jnp
import coax
import haiku as hk
from numpy import prod
import optax

name = 'ppo-pendulum'

# the Pendulum MDP
env = gym.make('Pendulum-v1')
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def func_pi(S, is_training):
	shared = hk.Sequential((
		hk.Linear(8), jax.nn.relu,
		hk.Linear(8), jax.nn.relu
	))
	mu = hk.Sequential((
		shared,
		hk.Linear(8), jax.nn.relu,
		hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
		hk.Reshape(env.action_space.shape)
	))
	logvar = hk.Sequential((
		shared,
		hk.Linear(8), jax.nn.relu,
		hk.Linear(prod(env.action_space.shape), w_init=jnp.zeros),
		hk.Reshape(env.action_space.shape)
	))
	return dict(mu=mu(S), logvar=logvar(S))


def func_v(S, is_training):
	seq = hk.Sequential((
		hk.Linear(8), jax.nn.relu,
		hk.Linear(8), jax.nn.relu,
		hk.Linear(8), jax.nn.relu,
		hk.Linear(1, w_init=jnp.zeros), jnp.ravel
	))
	return seq(S)


# define func approximators
pi = coax.Policy(func_pi, env)
v = coax.V(func_v, env)

# target network
pi_targ = pi.copy()

BUFFER_SIZE = 512
BATCH_SIZE = 32

# experience tracer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.9)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=BUFFER_SIZE)

# policy regularizer
policy_reg = coax.regularizers.EntropyRegularizer(pi, beta=0.01)

# updaters
simpletd = coax.td_learning.SimpleTD(v, optimizer=optax.adam(1e-3))
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=policy_reg, optimizer=optax.adam(1e-4))

# train
while env.T < int(1e6):
	s = env.reset()

	for t in range(env.spec.max_episode_steps):
		a, logp = pi_targ(s, return_logp=True)
		stp1, r, d, _ = env.step(a)

		# trace rewards
		tracer.add(s, a, r, d, logp)
		while tracer:
			buffer.add(tracer.pop())

		# learn
		if len(buffer) >= buffer.capacity:
			for _ in range(int(4 * buffer.capacity / BATCH_SIZE)):
				transition_batch = buffer.sample(batch_size=BATCH_SIZE)
				metrics_v, td_error = simpletd.update(transition_batch, return_td_error=True)
				metrics_pi = ppo_clip.update(transition_batch, td_error)
				env.record_metrics(metrics_v)
				env.record_metrics(metrics_pi)

			buffer.clear()  # on-policy so clear buffer
			pi_targ.soft_update(pi, tau=0.1)
		if d:
			break
		s = stp1

	# generate GIF to see what's happening
	if env.period(name='generate_gif', T_period=10000) and env.T > 5000:
		T = env.T - env.T % 10000
		coax.utils.generate_gif(env, policy=pi, filepath=f"./data/gifs/{name}/T{T:08d}.gif")
