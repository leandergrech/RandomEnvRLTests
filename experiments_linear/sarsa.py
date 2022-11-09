import os
import numpy as np
import warnings
from collections import deque
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions, get_reduced_discrete_actions
from experiments_linear.linear_q_function import QValueFunctionLinear, FeatureExtractor#, EfficientFeatureExtractor, QValueFunctionLinearEfficient
from utils.eval_utils import eval_agent

from tqdm import trange


def train_instance(**kwargs):
    """
    REDA training function. Kwargs holds all configurable training parameters and is saved to file
    :param kwargs: Many, many wonderful things in this dict
    :return: Absolutely nothing
    """
    '''
    ENVIRONMENT
    '''
    n_obs = kwargs.get('n_obs')
    n_act = kwargs.get('n_act')

    env = kwargs.get('env', None)
    # if env is None:
    #     env = REDA(n_obs, n_act)

    eval_env = REDA(n_obs, n_act, model_info=env.model_info)
    env_save_path = kwargs.get('env_save_path', None)
    if env_save_path:
        env.save_dynamics(env_save_path)

    actions = get_discrete_actions(n_act, 3)
    # actions = get_reduced_discrete_actions(n_act, 3)

    '''
    VALUE ESTIMATION
    '''
    feature_fn = FeatureExtractor(env)
    q = QValueFunctionLinear(feature_fn, actions)

    '''
    TRAINING PARAMETERS
    '''
    lr_fun = kwargs.get('lr_fun')
    exp_fun = kwargs.get('exp_fun')

    gamma = kwargs.get('gamma')
    nb_training_steps = kwargs.get('nb_training_steps')
    eval_every = kwargs.get('eval_every')
    eval_eps = kwargs.get('eval_eps')

    policy = kwargs.get('policy')

    '''
    TRAINING LOOP
    '''
    all_ep_lens = np.zeros(shape=(nb_training_steps//eval_every, eval_eps))
    all_returns = np.zeros_like(all_ep_lens)
    all_regrets = np.zeros_like(all_ep_lens)

    init_state_func = env.reset
    # objective_func = env.objective
    o = env.reset(init_state_func())
    a = q.greedy_action(o)

    for T in trange(nb_training_steps):
        otp1, r, d, info = env.step(actions[a])

        exploration = exp_fun(T)
        a_ = policy(otp1, q, exploration)

        if not d:
            target = r + gamma * q.value(otp1, a_)
        else:
            target = r

        lr = lr_fun(T)
        q.update(o, a, target, lr)

        if d:
            o = env.reset(init_state_func())
            a = policy(o, q, exploration)
        else:
            o = otp1.copy()
            a = a_

        if (T + 1) % eval_every == 0:
            ret = eval_agent(eval_env, q, eval_eps, init_func=init_state_func)
            idx = T // eval_every
            all_ep_lens[idx] = ret['ep_lens']
            all_returns[idx] = ret['returns']
            all_regrets[idx] = ret['regrets']

        if (T + 1) % kwargs['save_every'] == 0 or T == 0:
            save_dir = kwargs.get('results_path')
            save_dir = os.path.join(save_dir, 'q_func')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, f'q_step_{T+1}.pkl')
            q.save(save_path)

    return dict(ep_lens=all_ep_lens, returns=all_returns, regrets=all_regrets)


def quick_save(dir, step, q):
    save_dir = os.path.join(dir, 'q_func')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'q_step_{step + 1}.pkl')
    q.save(save_path)


def train_instance_early_termination(**kwargs):
    """
    REDA training function. Kwargs holds all configurable training parameters and is saved to file
    :param kwargs: Many, many wonderful things in this dict
    :return: Absolutely nothing
    """
    '''
    ENVIRONMENT
    '''
    n_obs = kwargs.get('n_obs')
    n_act = kwargs.get('n_act')

    env = kwargs.get('env', None)
    eval_env = kwargs.get('eval_env', None)

    # actions = get_discrete_actions(n_act, 3)
    actions = get_reduced_discrete_actions(n_act, 3)

    '''
    VALUE ESTIMATION
    '''
    feature_fn = FeatureExtractor(env)
    q = QValueFunctionLinear(feature_fn, actions)

    '''
    TRAINING PARAMETERS
    '''
    lr_fun = kwargs.get('lr_fun')
    exp_fun = kwargs.get('exp_fun')

    gamma = kwargs.get('gamma')
    nb_training_steps = kwargs.get('nb_training_steps')
    eval_every = kwargs.get('eval_every')
    _eval_every = eval_every
    eval_eps = kwargs.get('eval_eps')
    start_eval = kwargs.get('start_eval')

    policy = kwargs.get('policy')

    '''
    TRAINING LOOP
    '''
    save_path = kwargs.get('save_path')

    o = env.reset()
    a = q.greedy_action(o)

    nb_successes_early_termination = kwargs.get('nb_successes_early_termination')
    eval_successes = deque(maxlen=nb_successes_early_termination)
    T = 0
    for T in trange(nb_training_steps):
        otp1, r, d, info = env.step(actions[a])

        exploration = exp_fun(T)
        a_ = policy(otp1, q, exploration)

        if not d:
            target = r + gamma * q.value(otp1, a_)
        else:
            target = r

        lr = lr_fun(T)
        q.update(o, a, target, lr)

        if d:
            print(f'Success at step {T}')
            o = env.reset()
            a = policy(o, q, exploration)
        else:
            o = otp1.copy()
            a = a_

        if (T + 1) % eval_every == 0 and T >= start_eval:
            if max(np.ravel(np.abs(q.w))) > 100 or np.isnan(np.sum(q.w)):
                quick_save(save_path, T, q)
                q.reset_weights()
                print(f'\n\nState: {o}\n'
                      f'LR:  {lr:.2g}\n'
                      f'EXP: {exploration*100.0:.2f}%'
                      '\n******* RESET WEIGHTS *******\n\n')

            # print(f'Evaluating at step {T}')
            eval_ep_success = []

            # Run evaluation episodes
            for ep in range(eval_eps):
                eo = eval_env.reset()
                ed = False
                while not ed:
                    ea = q.greedy_action(eo)
                    eotp1, er, ed, info = eval_env.step(actions[ea])
                    eo = eotp1
                eval_ep_success.append(info['success'])

            # Store mean of eval episode successes
            eval_successes.append(np.mean(eval_ep_success)==1.0)

            if sum(eval_successes) > 0 and eval_every > 100:
                eval_every = 100
            else:
                eval_every = _eval_every

            if eval_successes[-1]:
                quick_save(kwargs.get('save_path'), T, q)
                # break
            if sum(eval_successes) == nb_successes_early_termination:
                break

        if (T + 1) % kwargs['save_every'] == 0 or T == 0:
            quick_save(save_path, T, q)

    return T





