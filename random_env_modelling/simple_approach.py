import os
import numpy as np
from datetime import datetime as dt
from collections import deque
import comet_ml
import torch as t
from torch.optim import SGD

from random_env.envs import RandomEnv
from my_agents_utils import MLP, get_writer
from constants import NB_TRAJS, DATA_DIR_NAME

COMET_WORKSPACE = "re_modelling"


def get_re_trajectories(n_obs, n_act):
    env = RandomEnv(n_obs=n_obs, n_act=n_act, estimate_scaling=False)
    env_dir = os.path.join(DATA_DIR_NAME, f'{n_obs}x{n_act}')
    env.load_dynamics(env_dir)

    data = np.load(os.path.join(env_dir, f'{NB_TRAJS}_{repr(env)}_trajectories.npz'))

    return env, data['obses'], data['acts'], data['rews'], data['obsestp1'], data['rets']


def test_models(vm, tm, env):
    # opt_env = RandomEnv(n_obs=env.obs_dimension, n_act=env.act_dimension, model_info=env.model_info)
    nb_eps = 100
    for max_steps in np.arange(1, 21):
        for ep in range(nb_eps):
            o_env = env.reset()
            o_model = np.copy(o_env)
            for step in range(max_steps):
                a_env = env.get_optimal_action(o_env)
                a_model = env.get_optimal_action(o_model)

                o_env, _, _, _ = env.step(a_env)
                o_model = tm(t.cat([o_model, a_model]))


RANDOM_SEED = 123
OPT_STEPS = 1000
RATIO_TEST = 0.2
EVAL_EVERY = 1
par_dir = 're_models'
session_name = f'sess_{dt.strftime(dt.now(), "%m%d%y_%H%M%S")}'
for n_obs in np.arange(2, 11, 2):
    for n_act in np.arange(2, 11, 2):
        env, obses, acts, rews, obsestp1, rets = get_re_trajectories(n_obs=n_obs, n_act=n_act)

        # Create models, loss fn and optimizers
        value_model = MLP(n_obs + n_act, 1, [100, 100], dtype=t.float64)
        transition_model = MLP(n_obs + n_act, n_obs, [100, 100], dtype=t.float64)

        loss_fn = t.nn.MSELoss()
        value_optim = SGD(value_model.parameters(), lr=1e-2)
        transition_optim = SGD(transition_model.parameters(), lr=1e-2)

        # Prepare logging
        experiment_name = f'{n_obs}x{n_act}_SEED{RANDOM_SEED}'
        writer = get_writer(experiment_name=experiment_name, project_name=session_name, workspace=COMET_WORKSPACE)

        # Prepare datasets
        value_in = t.cat([t.tensor(obses, dtype=t.float64), t.tensor(acts, dtype=t.float64)], dim=-1)
        value_out = t.tensor(rets, dtype=t.float64)
        transition_in = t.cat([t.tensor(obses, dtype=t.float64), t.tensor(acts, dtype=t.float64)], dim=-1)
        transition_out = t.tensor(obsestp1, dtype=t.float64)

        test_idxs = t.randperm(NB_TRAJS)[:int(RATIO_TEST * NB_TRAJS)]
        train_idxs = t.randperm(NB_TRAJS)[int(RATIO_TEST * NB_TRAJS):]

        value_in_train = value_in[train_idxs]
        value_out_train = value_out[train_idxs]
        transition_in_train = transition_in[train_idxs]
        transition_out_train = transition_out[train_idxs]
        value_in_test = value_in[test_idxs]
        value_out_test = value_out[test_idxs]
        transition_in_test = transition_in[test_idxs]
        transition_out_test = transition_out[test_idxs]

        print(f'SESSION {session_name} - EXP {experiment_name} - WS {COMET_WORKSPACE}')
        losses_v_train = deque(maxlen=OPT_STEPS)
        losses_t_train = deque(maxlen=OPT_STEPS)
        for i in range(OPT_STEPS):
            value_model.train()
            value_optim.zero_grad()
            pred_value_out = value_model(value_in_train)
            loss_v_train = loss_fn(pred_value_out, value_out_train)
            loss_v_train.backward()
            value_optim.step()
            losses_v_train.append(loss_v_train.item())

            transition_model.train()
            transition_optim.zero_grad()
            pred_obstp1 = transition_model(transition_in_train)
            loss_t_train = loss_fn(pred_obstp1, transition_out_train)
            loss_t_train.backward()
            transition_optim.step()
            losses_t_train.append(loss_t_train.item())

            loss_v_train, loss_t_train = np.mean(losses_v_train), np.mean(losses_t_train)
            writer.log_metrics(dict(value_loss_train=loss_v_train, transition_loss_train=loss_t_train), step=i)

            if i % EVAL_EVERY == 0:
                value_model.eval()
                transition_model.eval()

                pred_value_out = value_model(value_in_test)
                pred_obstp1 = transition_model(transition_in_test)

                loss_v_test = loss_fn(pred_value_out, value_out_test)
                loss_t_test = loss_fn(pred_obstp1, transition_out_test)

                writer.log_metrics(dict(value_loss_test=loss_v_test, transition_loss_test=loss_t_test),
                                   step=i)

                print(f'-> Iteration {i}/{OPT_STEPS}  {i / OPT_STEPS * 100}%')
                print(f' `-> Value Train loss: {loss_v_train}')
                print(f' `-> Value Test loss: {loss_v_test}')
                print(f' `-> Transition Train loss: {loss_t_train}')
                print(f' `-> Transition Test loss: {loss_t_test}')

                if loss_v_test > loss_v_train and loss_t_test > loss_v_train:
                    print('end - Finished training')
                    break
