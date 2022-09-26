import os
import numpy as np
import yaml
from datetime import datetime as dt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from random_env.envs import RandomEnv
from utils.training_utils import InitSolvableState
"""
This scipt is derived directly from the PPO training done in for the PhD on QFBEnv:
https://github.com/leandergrech/rl-qfb/blob/master/qfb_training.py
"""


class EvaluationAndCheckpointCallback(BaseCallback):
    MAX_EPS = 20

    def __init__(self, env, save_dir, EVAL_FREQ=100, CHKPT_FREQ=1000):
        self.env = env
        self.save_dir = save_dir
        self.model_name = os.path.split(save_dir)[-1]

        self.EVAL_FREQ = EVAL_FREQ
        self.CHKPT_FREQ = CHKPT_FREQ

        self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
        self.current_best_model_save_dir = ''

        self.gamma = 0.99
        self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]
        super(EvaluationAndCheckpointCallback, self).__init__()

    def quick_save(self, suffix=None):
        if suffix is None:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
        else:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

        self.model.save(save_path)
        if self.verbose > 0:
            print(f'Model saved to: {save_path}')

    def _on_step(self):
        if self.num_timesteps % self.EVAL_FREQ == 0:
            returns = []
            ep_lens = []
            success = []

            ### START OF EPISODE LOOP ###
            for ep in range(self.MAX_EPS):
                o = self.env.reset()

                ep_return = 0.0
                step = 0
                d = False
                while not d:
                    a = self.model.predict(o, deterministic=True)[0]
                    step += 1

                    o, r, d, _ = self.env.step(a)

                    ep_return += r

                ep_lens.append(step)
                returns.append(ep_return / self.env.REWARD_SCALE)
                if step < self.env.max_steps:
                    success.append(1.0)
                else:
                    success.append(0.0)
            ### END OF EPISODE LOOP ###

            returns = np.mean(returns)
            ep_lens = np.mean(ep_lens)
            success = np.mean(success) * 100

            ### SAVE SUCCESSFUL AGENTS ###
            if success > 0:
                self.quick_save()
                # if self.verbose > 1:
                #     print("Saving model checkpoint to {}".format(path))

            for tag, val in zip(('episode_return', 'episode_length', 'success'),
                                (returns, ep_lens, success)):
                self.logger.record(tag, val)
                # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                # self.locals['writer'].add_summary(summary, self.num_timesteps)

        if self.num_timesteps % self.CHKPT_FREQ == 0:
            self.quick_save()

        return True


class RESolvableInit(RandomEnv):
    def __init__(self, *args, **kwargs):
        super(RESolvableInit, self).__init__(*args, **kwargs)
        self.init_func = InitSolvableState(self, 0.9)

    def reset(self, init_state=None):
        if init_state is None:
            init_state = self.init_func()
        super(RESolvableInit, self).reset(init_state)
        return init_state


experiment_name = f"PPO_{dt.strftime(dt.now(), '%m%d%y_%H%M%S')}"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = RESolvableInit(16, 2, False)
env.save_dynamics(experiment_name)
eval_env = RESolvableInit(16, 2, False, model_info=env.model_info)

nb_steps = int(1e5)

for random_seed in (123, 234, 345, 456, 567):
    model_name = f"seed-{random_seed}"

    model_dir = os.path.join(experiment_name, model_name)

    save_dir = os.path.join(model_dir, 'saves')
    log_dir = os.path.join(model_dir, 'logs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(random_seed)

    ppo_params = dict(
        gamma=0.99,
        n_steps=128,
        batch_size=4,
        n_epochs=4,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs={'net_arch': [{'pi': [100, 100]},
                                      {'vf': [100, 100]}]},
        verbose=2,
        seed=random_seed,
        tensorboard_log=log_dir
    )
    model = PPO(PPOPolicy, env, **ppo_params)
    # eval_callback = EvaluationAndCheckpointCallback(eval_env, save_dir=save_dir,
    #                                                 EVAL_FREQ=100, CHKPT_FREQ=1000)
    callbacks = None #eval_callback
    with open(os.path.join(save_dir, 'training_params.yml'), 'w') as f:
        yaml.dump(ppo_params, f)

    model.learn(total_timesteps=nb_steps, log_interval=10, reset_num_timesteps=True,
                tb_log_name=model_name, callback=callbacks,
                eval_freq=100, eval_env=eval_env)

    save_path = os.path.join(save_dir, f'{model_name}_final')
    model.save(save_path)
    print(f'Model saved to: {save_path}')