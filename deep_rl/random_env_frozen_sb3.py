import os
import numpy as np
from datetime import datetime as dt
from collections import deque
import comet_ml

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch as t

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import TRPO
from random_env.envs import RandomEnv, RunningStats
from my_agents_utils import make_path_exist, get_writer, count_parameters


class EvalCheckpointEarlyStopTrainingCallback(BaseCallback):
    MAX_EPS = 20  # Number of evaluation episodes to run the latest policy
    END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES = 5  # self-explanatory
    SUCCESS_THRESHOLD = 100.0  # Any mean success rate during evaluation >= to this is considered total success

    def __init__(self, env, save_dir, EVAL_FREQ=100, CHKPT_FREQ=1000):
        """
        This callback automatically ends training after END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES number of
        consecutive total successes has elapsed. A total success is a success rate >= to SUCCESS_THRESHOLD.
        The init_callback(.) method needs to be called before anything is done with this callback. Here the agent
        object is passed so the callback can have access to its parameters.
        The on_step(.) method logs MAX_EPS trajectories using the agent's latest parameters to choose actions in env.
        :param env: OpenAI environment
        :param save_dir: Directory location where to store agent instances.
        :param EVAL_FREQ: Evaluate MAX_EPS episodes every this value steps.
        :param CHKPT_FREQ: Save the agent every this value steps.
        """
        self.env = env
        self.save_dir = save_dir
        self.model_name = os.path.split(save_dir)[-1]

        self.last_call_time = None
        self.last_call_step = None

        self.EVAL_FREQ = EVAL_FREQ
        self._eval_freq = EVAL_FREQ
        self.CHKPT_FREQ = CHKPT_FREQ

        self.current_best_model_ep_len = self.env.EPISODE_LENGTH_LIMIT
        self.current_best_model_save_dir = ''

        self.gamma = 0.99
        self.discounts = [np.power(self.gamma, i) for i in range(self.env.EPISODE_LENGTH_LIMIT)]
        self.successes = deque(
            maxlen=EvalCheckpointEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES)
        super(EvalCheckpointEarlyStopTrainingCallback, self).__init__()

        self.steps_since_last_animation = 0
        self.verbose = True

        self.success_history = deque(maxlen=self.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES)

    def __call__(self, *args, **kwargs):
        pass

    def quick_save(self, suffix=None):
        if suffix is None:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
        else:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

        if self.verbose > 1:
            print("Saving model checkpoint to {}".format(save_path))

        self.model.save(save_path)

    # if self.verbose > 0:
    # 	print(f'Model NOT saved to: {save_path}')

    def _on_step(self):
        self.num_timesteps = self.model.num_timesteps

        if self.num_timesteps % self.EVAL_FREQ == 0:
            returns = []
            ep_lens = []
            success = []
            rew_final_neg_init = []
            expected_rew_per_step = []

            observations = []
            actions = []
            trims = []

            ### START OF EPISODE LOOP ###
            for ep in range(self.MAX_EPS):
                ep_rewards = []

                o = self.env.reset()
                observations.append(o)

                ep_return = 0.0
                step = 0
                d = False
                while not d:
                    # a = self.model.predict(o, deterministic=True)[0]
                    a = self.model.predict(o, deterministic=True)
                    step += 1

                    o2, r, d, _ = self.env.step(a)

                    observations.append(np.copy(o2))
                    actions.append(a)
                    trims.append(o - o2)

                    o = o2
                    ep_return += r

                    ep_rewards.append(r)

                ep_lens.append(float(step))
                returns.append(ep_return)
                rew_final_neg_init.append(ep_rewards[-1] - ep_rewards[0])
                expected_rew_per_step.append(ep_return / step)
                if step < self.env.max_steps:
                    success.append(1.0)
                else:
                    success.append(0.0)
            ### END OF EPISODE LOOP ###

            returns = np.mean(returns)
            ep_lens = np.mean(ep_lens)
            success = np.mean(success)
            self.success_history.append(success == 1.0)
            success *= 100.0
            rew_final_neg_init = np.mean(rew_final_neg_init)
            expected_rew_per_step = np.mean(expected_rew_per_step)

            obs_mean = np.mean(observations)
            obs_std = np.mean(np.std(observations, axis=0))
            act_mean = np.mean(actions)
            act_std = np.mean(np.std(actions, axis=0))
            trim_mean = np.mean(trims)
            trim_std = np.mean(np.std(trims, axis=0))

            if self.last_call_time is None:
                self.last_call_time = dt.now()
                self.last_call_step = 0
                fps = 0
            else:
                this_call_time = dt.now()
                time_bet_calls = this_call_time - self.last_call_time
                steps_bet_calls = self.num_timesteps - self.last_call_step
                fps = steps_bet_calls / time_bet_calls.total_seconds()

                self.last_call_time = this_call_time
                self.last_call_step = self.num_timesteps

            if self.verbose:
                print(f'-> Training step: {self.num_timesteps}')
                print(f'\t-> Returns: {returns:.2f}')
                print(f'\t-> Episode length: {ep_lens:.2f}')
                print(f'\t-> Success rate: {success:.2f}')
                print(f'\t-> Rew. final - init: {rew_final_neg_init:.5f}')
                print(f'\t-> Expected rew. per step: {expected_rew_per_step}')
                print(f'\t-> Obs.  \u03BC = {obs_mean:.4f}, \u03C3 = {obs_std:.4f}')
                print(f'\t-> Act.  \u03BC = {act_mean:.4f}, \u03C3 = {act_std:.4f}')
                print(f'\t-> Trim. \u03BC = {trim_mean:.4f}, \u03C3 = {trim_std:.4f}')
                print(f'\t-> FPS = {fps:.2f}')
                print('')

            # Tensorboard
            # for tag, val in zip(('eval/episode_return', 'eval/episode_length', 'eval/success', 'eval/rew_final_neg_init',
            #                      'eval/expected_rew_per_step', 'train/fps'),
            #                     (returns, ep_lens, success, rew_final_neg_init, expected_rew_per_step, fps)):
            #     self.logger.record(tag, val)
            # Comet-ml
            self.logger.log_metrics({'eval/episode_return': returns,
                                     'eval/episode_length': ep_lens,
                                     'eval/success': success,
                                     'eval/rew_final_neg_init': rew_final_neg_init,
                                     'eval/expected_rew_per_step': expected_rew_per_step,
                                     'spaces/obs_mean': obs_mean, 'spaces/obs_std': obs_std,
                                     'spaces/act_mean': act_mean, 'spaces/act_std': act_std,
                                     'spaces/trim_mean': trim_mean, 'spaces/trim_std': trim_std,
                                     'train/fps': fps}, step=self.num_timesteps)

            # if sum(self.success_history) > 0 and self.EVAL_FREQ > 100:
            #     self.EVAL_FREQ = 100

            if sum(self.success_history) > 1 and self.success_history[-1]:
                if self.EVAL_FREQ > 1:
                    self.EVAL_FREQ = self.EVAL_FREQ // 2
            else:
                self.EVAL_FREQ = self._eval_freq

            if self.success_history[-1]:
                self.quick_save()

            if sum(self.success_history) == self.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES:
                self.quick_save()

                return False  # End training

        if self.num_timesteps % self.CHKPT_FREQ == 0:
            self.quick_save()

        return True


# COMET_WORKSPACE = 'testing-ppo-trpo'
COMET_WORKSPACE = 'testing-off-policy]'
# COMMON_ENV_DIR = '../identity_envs'
algo = 'TD3'

if 'PPO' in algo:
    # '''
    # HYPERPARAMETER GRID SEARCH PPO ON 5X5
    # '''
    # hparam_search_dict = dict(ent_coef=(0.01, 0.0),
    #                           vf_coef=(0.5, 1.0),
    #                           learning_rate=(1e-4, 1e-5),
    #                           n_steps=(100, 500),
    #                           batch_size=(64, 256))
    # keys, values = [], []
    # for k, v in hparam_search_dict.items():
    # 	keys.append(k)
    # 	values.append(v)
    # hparam_set = [{k: v for k, v in zip(keys, htuple)} for htuple in product(*values)]

    # for hparam_i, hparam_tuple in enumerate(hparam_set):
    # ppo_params.update(hparam_tuple)
    DEFAULT_PARAMS = dict(
        policy='MlpPolicy',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=- 1,
        target_kl=None,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device='auto',
        _init_setup_model=True)
elif 'TRPO' in algo:
    DEFAULT_PARAMS = dict(
        policy='MlpPolicy',
        learning_rate=0.001,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        cg_max_steps=15,
        cg_damping=0.1,
        line_search_shrinking_factor=0.8,
        line_search_max_iter=10,
        n_critic_updates=10,
        gae_lambda=0.95,
        use_sde=False,
        sde_sample_freq=- 1,
        normalize_advantage=True,
        target_kl=0.01,
        sub_sampling_factor=1,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device='auto',
        _init_setup_model=True
    )
elif 'TD3' in algo:
    DEFAULT_PARAMS = dict(
        policy='MlpPolicy',
        learning_rate=1e-3,
        buffer_size=1_000_000,  # 1e6
        learning_starts=100,
        batch_size=100,
        tau=5e-3,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        tensorboard_log=None,
        create_eval_env=False,
        policy_kwargs={'net_arch':[64, 64]},
        verbose=0,
        seed=None
    )

NB_STEPS = int(5e6)
EVAL_FREQ = 1000
CHKPT_FREQ = 50000

params = DEFAULT_PARAMS.copy()
session_name = dt.strftime(dt.now(), f'sess_{algo.lower()}_%m%d%y_%H%M%S')
# session_name = 'sess_trpo_102422_150542'
# session_name = f'sess_{algo.lower()}_050522_131337'
par_dir = os.path.join('sb3_identityenv_training', session_name)

for env_sz in np.arange(2, 3):
    N_OBS = env_sz
    N_ACT = env_sz

    for RANDOM_SEED in (123, 234, 345, 456, 567):
        '''Set seed to random generators'''
        t.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        params['seed'] = RANDOM_SEED

        '''Name agent (model) and create sub dirs required for saving and logging'''
        model_name = f'{algo}_' + f'IdentityEnv_{env_sz}obsx{env_sz}act' + f'_{RANDOM_SEED}seed'
        model_dir = os.path.join(par_dir, model_name)
        model_save_dir = os.path.join(model_dir, 'saves')
        model_log_dir = os.path.join(model_dir, 'logs')
        for fn in (model_save_dir, model_log_dir):
            if not os.path.exists(fn):
                os.makedirs(fn)
        params['tensorboard_log'] = model_log_dir

        '''Create environments with identity dynamics, duh'''
        env = RandomEnv(N_OBS, N_ACT, estimate_scaling=False)
        env.rm = np.diag(np.ones(N_OBS))
        env.pi = np.diag(np.ones(N_OBS))
        env.save_dynamics(model_dir)

        eval_env = RandomEnv(N_OBS, N_ACT, model_info=env.model_info)
        params['env'] = env


        '''Agent'''
        if 'PPO' in algo:
            model = PPO(**params)
        elif 'TRPO' in algo:
            model = TRPO(**params)
        elif 'TD3' in algo:
            model = TD3(**params)
        # print(f'-> Policy nb. of parameters: {count_parameters(model.actor)}')

        '''Callback evaluated agent every EVAL_FREQ steps and saved best model, and auto-saves every CHKPT_FREQ steps'''
        eval_callback = EvalCheckpointEarlyStopTrainingCallback(env=eval_env, save_dir=model_save_dir,
                                                                EVAL_FREQ=EVAL_FREQ, CHKPT_FREQ=CHKPT_FREQ)
        writer = get_writer(model_name, session_name, COMET_WORKSPACE)
        model.set_logger(writer)

        model.logger.log_parameters(params)

        '''Log some more info and save it in the same directory as the agent'''
        with open(os.path.join(model_dir, 'info.txt'), 'w') as f:
            '''Log hyperparameters used in this experiment'''
            f.write(f'-> {algo} parameters\n')
            for k, v in params.items():
                f.write(f'\t-> {k} = {v}\n')

        '''Start training'''
        model.learn(total_timesteps=NB_STEPS, callback=eval_callback)
