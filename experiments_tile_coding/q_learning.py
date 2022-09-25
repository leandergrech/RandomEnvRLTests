import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


from random_env.envs import RandomEnvDiscreteActions as REDA, get_discrete_actions
from tile_coding_re.tiles3_qfunction import Tilings, QValueFunctionTiles3
from utils.eval_utils import eval_agent, make_state_violins
from utils.training_utils import *#ExponentialDecay, get_training_utils_yaml_dumper

from tqdm import trange

# sns.set_context('paper')
sns.set_theme(style='whitegrid')


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

    env_save_path = kwargs.get('env_save_path', None)
    env = REDA(n_obs, n_act)
    eval_env = REDA(n_obs, n_act, model_info=env.model_info)

    if env_save_path:
        env.save_dynamics(env_save_path)

    actions = get_discrete_actions(n_act, 3)
    n_actions = len(actions)
    env_ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]

    '''
    VALUE ESTIMATION
    '''
    nb_tilings = kwargs.get('nb_tilings')
    nb_bins = kwargs.get('nb_bins')
    tilings = Tilings(nb_tilings, nb_bins, env_ranges, 2**20)
    q = QValueFunctionTiles3(tilings, n_actions)

    '''
    TRAINING PARAMETERS
    '''
    lr_fun = kwargs.get('lr_fun')
    exp_fun = kwargs.get('exp_fun')

    gamma = kwargs.get('gamma')
    nb_training_steps = kwargs.get('nb_training_steps')
    eval_every = kwargs.get('eval_every')

    o = env.reset()
    t = 0

    results_path = kwargs.get('results_path')

    '''
    TRAINING LOOP
    '''
    td_errors = np.zeros(nb_training_steps)
    eval_ep_lens = []
    iht_counts = []
    lrs = []
    ep_idx = 0
    for T in trange(nb_training_steps):
        if np.random.rand() < exp_fun(T):
            a = np.random.choice(n_actions)
        else:
            a = q.greedy_action(o)
        otp1, r, d, _ = env.step(actions[a])

        target = r + gamma * q.value(otp1, q.greedy_action(otp1))

        # td_errors[T] = q.update(o, a, target, lr_fun(T))
        lr = lr_fun(ep_idx)
        lrs.append(lr)
        td_errors[T] = q.update(o, a, target, lr)

        if d:
            t = 0
            o = env.reset()
            ep_idx += 1
        else:
            t += 1
            o = otp1

        if (T + 1) % eval_every == 0:
            init_obses, terminal_obses, ep_len_stats = eval_agent(eval_env, q, nb_eps=kwargs.get('eval_eps'))
            eval_ep_lens.append(ep_len_stats)
            iht_counts.append(tilings.count())
            make_state_violins(init_obses, terminal_obses, os.path.join(results_path, 'obses_violins', f'step-{T}.png'))

    return td_errors, eval_ep_lens, iht_counts, lrs


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)

    experiment_dir = dt.now().strftime('%m%d%y_%H%M%S')
    os.makedirs(experiment_dir)

    # lr_fun = ExponentialDecay(1.5e-1, 1000)
    # exp_fun = ExponentialDecay(1.0, 1000)
    # lr_fun = StepDecay(init=1e-2, decay_rate=.99, decay_every=1000)
    # exp_fun = StepDecay(init=1.0, decay_rate=.85, decay_every=1000)
    # lr_fun = CustomFunc()
    lr_fun = Constant(1e-2)
    exp_fun = Constant(0.1)

    train_params = dict(
        n_obs=2,
        n_act=2,
        nb_tilings=8,
        nb_bins=2,
        env_save_path=experiment_dir,
        results_path=experiment_dir,
        lr_fun=lr_fun,
        exp_fun=exp_fun,
        nb_training_steps=10000,
        eval_every=500,
        eval_eps=20,
        gamma=0.99
    )
    with open(os.path.join(experiment_dir, "train_params.yml"), "w") as f:
        f.write(yaml.dump(train_params, Dumper=get_training_utils_yaml_dumper()))

    errors, eval_ep_lens, iht_counts, lrs = train_instance(**train_params)

    el_means = [item['mean'] for item in eval_ep_lens]
    el_mins = [item['min'] for item in eval_ep_lens]
    el_maxs = [item['max'] for item in eval_ep_lens]

    fig, axs = plt.subplots(4, 1, gridspec_kw=dict(height_ratios=[3, 3, 3, 3]))
    xrange = np.arange(len(errors))
    eval_xrange = np.linspace(0, train_params['nb_training_steps'], len(el_means))
    axs[0].plot(xrange, errors)
    axs[0].set_ylabel('TD error')
    axs[1].plot(eval_xrange, el_mins, ls=':')
    axs[1].plot(eval_xrange, el_maxs, ls=':')
    axs[1].plot(eval_xrange, el_means)
    axs[1].set_ylabel('Eval ep lens')
    axs[2].plot(eval_xrange, iht_counts)
    axs[2].set_ylabel('Nb tiles')

    ax = axs[-1]
    ax_lr = ax
    ax_exp = ax.twinx()
    lr_line, = ax_lr.plot(xrange, lrs, c='k', ls=':')
    exp_line, = ax_exp.plot(xrange, [exp_fun(i) for i in xrange], c='r')
    ax_lr.legend(handles=[lr_line, exp_line], labels=['Learning rate', 'Exploration rate'])

    ax_exp.spines['right'].set_color('r')
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'experiment_overview.png'))

    plt.show()



