import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorlayer.optimizers import AMSGrad

from rlzoo.algorithms import TRPO
from rlzoo.common.value_networks import ValueNetwork
from rlzoo.common.policy_networks import StochasticPolicyNetwork

from gym.spaces import Dict as GymDict

from random_env.envs import RandomEnv

n_obs, n_act = 5, 5
env = RandomEnv(n_obs, n_act, estimate_scaling=True)

# state_space = GymDict({'state': env.observation_space})
state_space = env.observation_space

critic_model = ValueNetwork(state_space=state_space,
                            hidden_dim_list=[50, 50])
policy_model = StochasticPolicyNetwork(state_space=state_space,
                                       action_space=env.action_space,
                                       hidden_dim_list=[50, 50])

optimizers_list = [AMSGrad(learning_rate=1e-2)]

agent = TRPO([critic_model, policy_model], optimizers_list)
agent.learn(env, train_episodes=500, test_episodes=100, max_steps=100, save_interval=100,
            gamma=0.9, mode='train', render=False, batch_size=32, backtrack_iters=10, backtrack_coeff=0.8,
            train_critic_iters=80, plot_func=None)
