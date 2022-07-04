import gym
import jax
import coax
import haiku as hk
import jax.numpy as jnp
from optax import adam

name = 'ppo'

env = gym.make('PongNoFrameskip-v4')
env = gym.wrappers.AtariPreprocessing(env)
env = coax.wrappers.FrameStacking(env, num_frames=3)
env = coax.wrappers.TrainMonitor(env, name=name, tensorboard_dir=f"./data/tensorboard/{name}")


def shared(S, is_training):
	seq = hk.Sequential([
		coax.utils.diff_transform,
		hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
		hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
		hk.Flatten(),
	])
	X = jnp.stack(S, axis=-1) / 225.  # stack frames
	return seq(X)


def func_pi(S, is_training):
	logits = hk.Sequential((
		hk.Linear(256), jax.nn.relu,
		hk.Linear(env.action_space.n, w_init=jnp.zeros)
	))
	X = shared(S, is_training)
	return {'logits': logits(X)}


def func_v(S, is_training):
	value = hk.Sequential((
		hk.Linear(256), jax.nn.relu,
		hk.Linear(1, w_init=jnp.zeros), jnp.ravel
	))
	X = shared(S, is_training)
	return value(X)


# function approximators
pi = coax.Policy(func_pi, env)
v = coax.V(func_v, env)

# target networks
pi_behaviour = pi.copy()
v_targ = v.copy()

# policy regularizer (avoid premature exploitation)
entropy = coax.regularizers.EntropyRegularizer(pi, beta=0.001)

# updaters
simpletd = coax.td_learning.SimpleTD(v, v_targ, optimizer=adam(3e-4))
ppo_clip = coax.policy_objectives.PPOClip(pi, regularizer=entropy, optimizer=adam(3e-4))

BUFFER_SIZE = 128
NB_EPOCHS = 4
BATCH_SIZE = 32

# reward tracer and replay buffer
tracer = coax.reward_tracing.NStep(n=5, gamma=0.99)
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=BUFFER_SIZE)

# run episodes
while env.T < int(3e6):
	s = env.reset()
	for t in range(env.spec.max_episode_steps):
		a, logp = pi_behaviour(s, return_logp=True)
		stp1, r, d, _ = env.step(a)

		# trace rewards and add transition to replay buffer
		tracer.add(s, a, r, d, logp)
		while tracer:
			buffer.add(tracer.pop())

		# learn
		if len(buffer) >= buffer.capacity:
			num_batches = int(NB_EPOCHS * buffer.capacity / BATCH_SIZE)  # NB_EPOCHS epochs per loop
			for _ in range(num_batches):
				transition_batch = buffer.sample(BATCH_SIZE)
				metrics_v, td_error = simpletd.update(transition_batch, return_td_error=True)
				metrics_pi = ppo_clip.update(transition_batch, td_error)
				env.record_metrics(metrics_v)
				env.record_metrics(metrics_pi)
			# since on-policy, clear buffer
			buffer.clear()

			# sync target networks
			pi_behaviour.soft_update(pi, tau=0.1)
			v_targ.soft_update(v, tau=0.1)

		if d:
			break
		s = stp1

	# generate an animated GIF to see what's happening
	if env.period(name='generate_gif', T_period=10000) and env.T > 50000:
		T = env.T - env.T % 10000  # round to 10000s
		coax.utils.generate_gif(
			env=env, policy=pi, resize_to=(320, 420),
			filepath=f"./data/gifs/{name}/T{T:08d}.gif")
