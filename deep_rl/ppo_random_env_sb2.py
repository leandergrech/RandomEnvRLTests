import os
import numpy as np
from copy import deepcopy
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime as dt
import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import MlpPolicy as PPOPolicy

from random_env import RandomEnv


class EvalCheckptEarlyStopTrainingCallback(BaseCallback):
    MAX_EPS = 20  # Number of evaluation episodes to run the latest policy
    END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES = 7  # self-explanatory
    SUCCESS_THRESHOLD = 90.0  # Any mean success rate during evaluation greater than this is considered total success

    def __init__(self, env, save_dir, EVAL_FREQ=100, CHKPT_FREQ=1000):
        """
        This callback automatically ends training after
        :param env:
        :param save_dir:
        :param EVAL_FREQ:
        :param CHKPT_FREQ:
        """
        self.env = env
        self.save_dir = save_dir
        self.model_name = os.path.split(save_dir)[-1]

        self.EVAL_FREQ = EVAL_FREQ
        self.CHKPT_FREQ = CHKPT_FREQ

        self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
        self.current_best_model_save_dir = ''

        self.gamma = 0.99
        self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]
        self.successes = deque(maxlen=EvalCheckptEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES)
        super(EvalCheckptEarlyStopTrainingCallback, self).__init__()

    def quick_save(self, suffix=None):
        if suffix is None:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
        else:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

        if self.verbose > 1:
            print("Saving model checkpoint to {}".format(save_path))

        self.model.save(save_path)
        if self.verbose > 0:
            print(f'Model saved to: {save_path}')

    def _on_step(self):
        if self.num_timesteps % self.EVAL_FREQ == 0:
            returns = []
            ep_lens = []
            success = []

            observations = []
            actions = []

            ### START OF EPISODE LOOP ###
            for ep in range(self.MAX_EPS):
                o = self.env.reset()
                observations.append(o)

                ep_return = 0.0
                step = 0
                d = False
                while not d:
                    a = self.model.predict(o, deterministic=True)[0]
                    step += 1

                    o, r, d, _ = self.env.step(a)

                    observations.append(o)
                    actions.append(a)

                    ep_return += r

                ep_lens.append(step)
                returns.append(ep_return)
                if step < self.env.max_steps:
                    success.append(1.0)
                else:
                    success.append(0.0)
            ### END OF EPISODE LOOP ###

            returns = np.mean(returns)
            ep_lens = np.mean(ep_lens)
            success = np.mean(success) * 100.0
            obs_mean = np.mean(observations)
            obs_std = np.std(observations)
            act_mean = np.mean(actions)
            act_std = np.std(actions)

            for tag, val in zip(('episode_return', 'episode_length', 'success', 'spaces/obs_mean', 'spaces/obs_std',
                                 'spaces/act_mean', 'spaces/act_std'),
                                (returns, ep_lens, success, obs_mean, obs_std, act_mean, act_std)):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)

            ### SAVE SUCCESSFUL AGENTS ###
            if success > 0:
                self.quick_save()
                self.successes.append(success >= EvalCheckptEarlyStopTrainingCallback.SUCCESS_THRESHOLD)
                if sum(self.successes) >= EvalCheckptEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES:
                    return False  # End training

        if self.num_timesteps % self.CHKPT_FREQ == 0:
            self.quick_save()

        if self.num_timesteps % 2000 == 0:
            self.animation()

        return True

    def animation(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(f'Evaluation at training step: {self.num_timesteps}')
        plt.ion()

        o1 = self.env.reset()

        for ax in fig.axes:
            ax.axhline(0.0, color='k', ls='--')
            ax.grid(which='both', color='gray')

        oline, = ax1.plot(o1)
        ax1.set_title('Observation')

        aline, = ax2.plot(np.zeros(self.env.act_dimension))
        ax2.set_title('Action')

        fig.tight_layout()

        d = False
        new_ylim = lambda ax, ydat: (np.min(np.concatenate([ax.get_ylim(), ydat])),
                                     np.max(np.concatenate([ax.get_ylim(), ydat])))
        while not d:
            a = self.model.predict(o1, deterministic=True)[0]
            o2, r, d, _ = self.env.step(a)

            oline.set_ydata(o2)
            aline.set_ydata(a)

            ax1.set_ylim(new_ylim(ax1, o2))
            ax2.set_ylim(new_ylim(ax1, a))
            plt.pause(0.01)
            o1 = o2
        plt.pause(2)
        plt.close()


ppo_params = dict(
    gamma=0.99,
    n_steps=512,
    ent_coef=0.01,
    learning_rate=2.5e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    lam=0.95,
    nminibatches=4,
    noptepochs=4,
    cliprange=0.2,
    cliprange_vf=None,
    policy_kwargs={'net_arch': [50, 50]}
)
random_seed = 123
nb_steps = int(5e5)
par_dir = os.path.join('param_random_env_results', dt.strftime(dt.now(), 'training_session_%m%d%y_%H%M%S'))
if not os.path.exists(par_dir):
    os.makedirs(par_dir)

# for env_sz in np.arange(10, 101, 10):

n_obs = 5
n_act = 5
env = RandomEnv(n_obs=n_obs, n_act=n_act, estimate_scaling=True)
print(env.trim_stats)

eval_env = deepcopy(env)

model_name = f'PPO_random_env_{n_obs}x{n_act}'
log_dir = os.path.join(par_dir, 'logs')
save_dir = os.path.join(par_dir, model_name)

for path in (log_dir, save_dir):
    if not os.path.exists(path):
        os.makedirs(path)

model = PPO2(PPOPolicy, env, **ppo_params, verbose=1,
             tensorboard_log=log_dir, _init_setup_model=True,
             full_tensorboard_log=False, seed=random_seed, n_cpu_tf_sess=None)

eval_callback = EvalCheckptEarlyStopTrainingCallback(eval_env, save_dir=save_dir,
                                                     EVAL_FREQ=100, CHKPT_FREQ=1000)

with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
    f.write('-> PPO parameters\n')
    for k, v in ppo_params.items():
        f.write(f'\t-> {k} = {v}\n')
    f.write(f'-> Info: Environment estimated scaling:\n\t{eval_env.trim_stats}\n')
    f.write(f'-> Info: Environment model output normalised\n')
model.learn(total_timesteps=nb_steps, log_interval=10, reset_num_timesteps=True,
            tb_log_name=model_name, callback=eval_callback)
