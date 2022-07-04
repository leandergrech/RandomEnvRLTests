from comet_ml import Experiment

# comet_ml.init(project_name='tensorboardX')
# from tensorboardX import SummaryWriter
COMET_API_KEY = "LvCyhW3NX1yaPPqv3LIMb1qDr"

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from collections import deque
from functools import partial

import torch as t
from torch import nn
import torch.nn.functional as F


def timeit(func, inputs, number=100):
    start = dt.now()
    f = partial(func, *inputs)
    for _ in range(number):
        out = f()
    end = dt.now()

    return end - start, out


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, act_fun=None, dtype=None):
        """
        Instantiates a 1D multi-layered perceptron with dense connections. Multi-layer with hidden functions passed
        through act_fun and a linear last layer.
        :param in_dim: Input dimension
        :param out_dim: Output dimension
        :param hidden_layers: List of integers - size of each hidden layer
        :param act_fun: Activation function for hidden layers
        """
        super(MLP, self).__init__()
        if act_fun is None:
            self.act_fun = F.relu
        else:
            self.act_fun = act_fun
        if dtype is None:
            dtype = t.float
        self.dtype = dtype

        layer_input_sizes = np.concatenate([[in_dim], hidden_layers])
        layer_output_sizes = np.concatenate([hidden_layers, [out_dim]])
        layers = []
        for in_sz, out_sz in zip(layer_input_sizes, layer_output_sizes):
            layers.append(nn.Linear(in_sz, out_sz, dtype=dtype))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if not isinstance(x, t.Tensor):
            x = t.tensor(x, dtype=self.dtype)

        for l in self.layers[:-1]:
            x = self.act_fun(l(x))

        return self.layers[-1](x)


# from stable_baselines3.common.callbacks import BaseCallback
class EvalCheckpointEarlyStopTrainingCallback():
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

    def __call__(self, *args, **kwargs):
        pass

    def quick_save(self, suffix=None):
        if suffix is None:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{self.num_timesteps}_steps')
        else:
            save_path = os.path.join(self.save_dir, f'{self.model_name}_{suffix}')

        if self.verbose > 1:
            print("Saving model checkpoint to {}".format(save_path))

        # self.model.save(save_path)
        if self.verbose > 0:
            print(f'Model NOT saved to: {save_path}')

    def init_callback(self, model, writer):
        self.model = model
        self.writer = writer

    def on_step(self):
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
            success = np.mean(success) * 100.0
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

            self.writer.log_metrics({'eval/episode_return': returns,
                                     'eval/episode_length': ep_lens,
                                     'eval/success': success,
                                     'eval/rew_final_neg_init': rew_final_neg_init,
                                     'eval/expected_rew_per_step': expected_rew_per_step,
                                     'spaces/obs_mean': obs_mean, 'spaces/obs_std': obs_std,
                                     'spaces/act_mean': act_mean, 'spaces/act_std': act_std,
                                     'spaces/trim_mean': trim_mean, 'spaces/trim_std': trim_std,
                                     'train/fps': fps}, step=self.num_timesteps)

            ### SAVE SUCCESSFUL AGENTS ###
            if success > 50.0:
                self.quick_save()
                if success >= EvalCheckpointEarlyStopTrainingCallback.SUCCESS_THRESHOLD:
                    self.successes.append(1)
                else:
                    self.successes.clear()

                if len(self.successes) >= EvalCheckpointEarlyStopTrainingCallback.END_TRAINING_AFTER_N_CONSECUTIVE_SUCCESSES:
                    return False  # End training

        if self.num_timesteps % self.CHKPT_FREQ == 0:
            self.quick_save()

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

            oline.set_ydata(np.copy(o2))
            aline.set_ydata(np.copy(a))

            ax1.set_ylim(new_ylim(ax1, o2))
            ax2.set_ylim(new_ylim(ax2, a))
            plt.pause(0.01)
            o1 = o2
        plt.pause(2)
        plt.close()


class ExperimentWrapper(Experiment):
    def __init__(self, *args, **kwargs):
        super(ExperimentWrapper, self).__init__(*args, **kwargs)

    def record(self, tag, val, step=None, **kwargs):
        if 'exclude' in kwargs:
            return None
        return self.log_parameters({tag: val}, step=step)

    def dump(self, *args, **kwargs):
        pass


def get_writer(experiment_name, project_name, workspace):
    exp = ExperimentWrapper(api_key=COMET_API_KEY, project_name=project_name, workspace=workspace)
    exp.add_tag(experiment_name)
    return exp


def make_path_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def compute_returns(r_traj, DISCOUNT=0.99):
    r_traj = t.tensor(r_traj, dtype=t.double)
    k = len(r_traj)
    pows = t.pow(t.tensor(DISCOUNT, dtype=t.double), t.arange(k))
    discounted_rews = pows * t.cat([t.tensor(r_traj[1:]), t.tensor([0.0], dtype=t.float64)], dim=-1)
    discounted_rets = t.cumsum(discounted_rews.flip(-1), dim=-1).flip(-1) / pows

    return discounted_rets.detach().numpy()
