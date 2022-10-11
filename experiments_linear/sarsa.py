import os
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from linear_q_function import QValueFunctionLinear, FeatureExtractor
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

    policy = kwargs.get('policy')

    '''
    TRAINING LOOP
    '''
    all_ep_lens = []
    all_returns = []
    all_regrets = []

    init_state_func = env.reset
    objective_func = env.objective
    o = env.reset(init_state_func())
    a = q.greedy_action(o)

    for T in trange(nb_training_steps):
        otp1, _, d, info = env.step(actions[a])
        r = objective_func(otp1)
        if info['success']:
            r = 0

        exploration = exp_fun(T)
        a_ = policy(otp1, q, exploration)

        target = r + gamma * q.value(otp1, a_)

        # td_errors[T] = q.update(o, a, target, lr_fun(T))
        lr = lr_fun(T)
        q.update(o, a, target, lr)

        if d:
            o = env.reset(init_state_func())
            a = policy(o, q, exploration)
        else:
            o = otp1.copy()
            a = a_

        if (T + 1) % eval_every == 0:
            ret = eval_agent(eval_env, q, kwargs.get('eval_eps'), init_func=init_state_func)
            all_ep_lens.append(ret['ep_lens'])
            all_returns.append(ret['returns'])
            all_regrets.append(ret['regrets'])

        if (T + 1) % kwargs['save_every'] == 0 or T == 0:
            save_dir = kwargs.get('results_path')
            save_dir = os.path.join(save_dir, 'q_func')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join(save_dir, f'q_step_{T+1}.pkl')
            q.save(save_path)

    return dict(ep_lens=all_ep_lens, returns=all_returns, regrets=all_regrets)





