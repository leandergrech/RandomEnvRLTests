import os
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from tqdm import tqdm as pbar
from tile_coding_re.tiles3_qfunction import QValueFunctionTiles3
from random_env.envs.random_env_discrete_actions import RandomEnvDiscreteActions as REDA, get_discrete_actions, REDAClip
from tile_coding_re.heatmap_utils import make_heatmap
from experiments_tile_coding.eval_utils import play_episode

def get_latest_experiment():
    lab_dir = '/home/leander/code/RandomEnvRLTests/experiments_tile_coding/'
    experiments = []
    for fn in os.listdir(lab_dir):
        if 'sarsa' in fn:
            experiments.append(fn)
    experiments = sorted(experiments)

    experiment_name = experiments[-1]

    return os.path.join(lab_dir, experiment_name)


from eval_utils import get_q_func_filenames, get_q_func_xrange, get_val, get_q_func_step


def plot_q_vals_grid_tracking_states():
    experiment_dir = get_latest_experiment()
    experiment_name = os.path.split(experiment_dir)

    env = REDA.load_from_dir(experiment_dir)
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    q_func_filenames = get_q_func_filenames(experiment_dir)
    nb_q_funcs = len(q_func_filenames)
    xrange = get_q_func_xrange(q_func_filenames)

    # Initialise grid tracking states
    n_tracking_dim = 5
    tracking_ranges = [[l, h] for l, h in zip(env.observation_space.low, env.observation_space.high)]
    tracking_states = np.array([list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])
    nb_tracked = len(tracking_states)

    vals = np.zeros(shape=(nb_tracked, nb_q_funcs))
    for j, qfn in enumerate(q_func_filenames):
        q = QValueFunctionTiles3.load(qfn)
        for i, ts in enumerate(tracking_states):
            vals[i,j] = get_val(q, ts, nb_actions)


    fig, axs = plt.subplots(2, gridspec_kw=dict(height_ratios=[1,2]))
    cmap = mpl.cm.get_cmap('tab10')
    cmap_x = np.linspace(0, 1, nb_tracked)

    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], marker='x', c=cmap(cmap_x[i]))

    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    for i, val in enumerate(vals):
        ax.plot(xrange, val, c=cmap(cmap_x[i]), lw=1.0)

    fig.suptitle(experiment_name)
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'tracked_vals.png'))
    plt.show()


def circular_normal_sample(r, s):
    if r > 0:
        R = np.random.normal(r, s)
        theta = 2 * np.pi * np.random.rand()
        return [R * np.cos(theta), R * np.sin(theta)]
    else:
        return np.random.normal(r, s, 2)


def nball_uniform_sample(dim, rlow, rhigh):
    X = np.random.uniform(-1, 1, dim)
    R = np.sqrt(np.sum(np.square(X))) # Sampled point lies on n-ball with radius R
    X /= R # Now point lies on unit n-ball surface

    M = np.random.uniform(rlow, rhigh) # Magnitude sampled from uni. dist. to get rlow-rhigh band
    return X * M
    #
    # theta = 2 * np.pi * np.random.rand()
    # return [R * np.cos(theta), R * np.sin(theta)]


def plot_q_vals_region_sampling_tracking_states(experiment_dir):
    env = REDAClip.load_from_dir(experiment_dir)
    n_obs = env.obs_dimension
    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    print('Searching for training data')
    q_func_filenames = get_q_func_filenames(experiment_dir)
    xrange = get_q_func_xrange(q_func_filenames)

    print('Creating tracking regions')
    nb_samples_per_region = 200
    nb_regions = 5
    rstep = 1 / nb_regions

    tracking_states = []
    for r in np.arange(nb_regions):
        for i in range(nb_samples_per_region):
            ts = nball_uniform_sample(n_obs, rlow=r * rstep, rhigh=r * rstep + rstep)
            tracking_states.append(ts)
    nb_tracked = len(tracking_states)

    print('Estimating values at tracked states')
    tracked_vals = [[] for _ in range(nb_tracked)] # shape=(nb_tracked, len(q_func_filenames))
    for qfn in pbar(q_func_filenames):
        q = QValueFunctionTiles3.load(qfn)
        for i, ts in enumerate(tracking_states):
            tracked_vals[i].append(get_val(q, ts, nb_actions))

    print('Plotting')
    fig, axs = plt.subplots(2, figsize=(15,10))
    fig.suptitle(experiment_dir)
    cmap = mpl.cm.get_cmap('jet')
    region_color = lambda i: cmap((i // nb_samples_per_region)
                                  * nb_samples_per_region/(nb_tracked - nb_samples_per_region)) # Separates regions by colour
    ax = axs[0]
    ax.set_xlabel('State dimension 0')
    ax.set_ylabel('State dimension 1')
    for i, ts in enumerate(tracking_states):
        ax.scatter(ts[0], ts[1], c=region_color(i), marker='x')

    ax = axs[1]
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Estimated values')
    for val_idx in np.arange(0, nb_tracked, nb_samples_per_region):
        vals = tracked_vals[val_idx:val_idx+nb_samples_per_region]  # split per region
        vals_min = np.min(vals, axis=0)
        vals_max = np.max(vals, axis=0)
        vals_mean = np.mean(vals, axis=0)
        vals_std = np.std(vals, axis=0)
        vals_lower = vals_mean - vals_std
        vals_upper = vals_mean + vals_std
        label = 'Mean' if val_idx == 0 else None
        ax.plot(xrange, vals_mean, c=region_color(val_idx), lw=1.0, label=label)
        c = region_color(val_idx)
        label = '$min\\rightarrow -\sigma$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_min, vals_lower, edgecolor=c, facecolor='None', hatch='//', label=label, alpha=0.4)
        label = '$-\sigma\\rightarrow\sigma$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_lower, vals_upper, facecolor=c, alpha=0.2, label=label)
        label = '$\sigma\\rightarrow max$' if val_idx == 0 else None
        ax.fill_between(xrange, vals_upper, vals_max, edgecolor=c, facecolor='None', hatch='\\\\', label=label, alpha=0.4)
    ax.legend(loc='best')
    plt.savefig(os.path.join(experiment_dir, 'tracked_vals_per_region.png'))
    plt.show()


def plot_individual_action_advs(experiment_dir, query_step):
    # experiment_dir = get_latest_experiment()
    # experiment_dir = 'changing_policy_type/sarsa_091322_220631_0/boltz-5'
    experiment_name = os.path.split(experiment_dir)

    # env = REDA.load_from_dir(experiment_dir)
    env = REDAClip(2, 2, state_clip=0.0)
    env.load_dynamics(os.path.join(experiment_dir, 'REDAClip_0.0clip_2obsx2act_dynamics.pkl'))
    env.state_clip = 0.0

    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    print('Searching for training data')
    q_func_filenames = get_q_func_filenames(experiment_dir)
    training_steps = [get_q_func_step(qfn) for qfn in q_func_filenames]

    closest_training_step_idx = np.argmin(np.abs(np.subtract(training_steps, query_step)))
    qfn_at_step = q_func_filenames[closest_training_step_idx]
    at_training_step = get_q_func_step(qfn_at_step)

    # Initialise grid tracking states
    n_tracking_dim = 64
    tracking_lim = 1.2
    tracking_ranges = [[-tracking_lim, -tracking_lim], [tracking_lim, tracking_lim]]
    tracking_ranges = [[l, h] for l, h in zip(*tracking_ranges)]
    tracking_states = np.array(
        [list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])

    # Load Q-table
    q = QValueFunctionTiles3.load(qfn_at_step)

    # Play test episodes to superimpose on heatmaps
    nb_test_eps = 3
    test_obses = []
    test_acts = []
    for _ in range(nb_test_eps):
        obses, acts, _ = play_episode(env, q, nball_uniform_sample(0.8, 0.9))
        if len(obses)>50:
            test_obses.append(obses)
            test_acts.append(acts)
            if len(test_obses)>10:
                break

    # Estimate and store state values
    tracking_estimated_vals = []
    for ts in tracking_states:
        tracking_estimated_vals.append(np.mean([q.value(ts, a_) for a_ in range(nb_actions)]))

    # Plot the fuckin shit already
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = np.ravel(axs)
    min_adv = np.inf
    max_adv = -np.inf
    ims = []
    for action_idx, (ax, act) in enumerate(zip(axs, actions)):
        init_state = np.array([0.0, 0.0])
        env.reset(init_state.copy())
        otp1, *_ = env.step(act)
        ax.plot(*np.vstack([init_state, otp1]).T, c='k', marker='x', zorder=20)

        # Plot test episodes
        for tobses, tacts in zip(test_obses, test_acts):
            tobses = np.array(tobses).T
            ax.plot(tobses[0], tobses[1], c='k', zorder=25)

        # Calculate advantages
        tracked_advs = []
        for ts, tv in zip(tracking_states, tracking_estimated_vals):
            # ax.scatter(ts[0], ts[1], marker='o', c='k')
            qval = q.value(ts, action_idx)
            adv = qval - tv
            tracked_advs.append(adv)

        # For clim to have the same range
        min_adv = np.min([min_adv, *tracked_advs])
        max_adv = np.max([min_adv, *tracked_advs])

        # Magic fuckery to align array to correct heatmap orientation
        tracked_advs = np.array(tracked_advs).reshape((n_tracking_dim, n_tracking_dim))
        tracked_advs = np.flipud(np.rot90(tracked_advs))

        # Heatmap plotting
        im = make_heatmap(ax, tracked_advs, tracking_states.T[0], tracking_states.T[1], title=f'{act}')
        ims.append(im)
        ax.add_patch(mpl.patches.Circle((0,0), env.GOAL, edgecolor='g', ls='--', facecolor='None', zorder=20))

    for im in ims:
        im.set_clim((min_adv, max_adv))

    fig.suptitle(f'Dir = {exp_dir}\nEnv = {repr(env)}\n'
                 f'Training step = {at_training_step}')
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.88, wspace=0.083, hspace=0.321)
    save_path = os.path.join(experiment_dir, f'individual_action_advs_at-step-{at_training_step}.png')
    plt.savefig(save_path)
    print(f'Saved figure to: {save_path}')

    # fig, ax = plt.subplots()
    # qvals = []
    # for ts in tracking_states:
    #     qvals.append(np.max([q.value(ts, a_) for a_ in range(nb_actions)]))
    # qvals = np.array(qvals).reshape((n_tracking_dim, n_tracking_dim))
    # qvals = np.flipud(np.rot90(qvals))
    # make_heatmap(ax, qvals, tracking_states.T[0], tracking_states.T[1], title='Estimated values')
    # ax.add_patch(mpl.patches.Circle((0, 0), env.GOAL, edgecolor='g', ls='--', facecolor='None', zorder=20))
    # plt.savefig(os.path.join(experiment_dir, f'vals_at-step-{at_training_step}.png'))
    # # plt.show()


def plot_compare_qvals_during_training():
    experiment_dir = 'changing_policy_type/sarsa_091322_005443_0/eps-greedy'
    experiment_name = os.path.split(experiment_dir)

    # env = REDA.load_from_dir(experiment_dir)
    env = REDAClip(2, 2, state_clip=0.0)
    env.load_dynamics(os.path.join(experiment_dir, 'REDAClip_0.0clip_2obsx2act_dynamics.pkl'))
    env.state_clip = 0.0

    actions = get_discrete_actions(env.act_dimension, 3)
    nb_actions = len(actions)

    print('Searching for training data')
    q_func_filenames = get_q_func_filenames(experiment_dir)
    training_steps = [get_q_func_step(qfn) for qfn in q_func_filenames]
    # qs = [QValueFunctionTiles3.load(item) for item in q_func_filenames]
    nb_q_funcs = len(q_func_filenames)

    # query_training_step_start = 79500 - 1
    # query_training_step_finish = 80000- 1
    query_training_step_start = 80000 - 1
    query_training_step_finish = 80500 - 1

    qs = []
    actual_steps = []
    for step in (query_training_step_start, query_training_step_finish):
        qfn_at_step = q_func_filenames[np.searchsorted(training_steps, step)]
        at_training_step = get_q_func_step(qfn_at_step)
        actual_steps.append(at_training_step)
        q = QValueFunctionTiles3.load(qfn_at_step)
        qs.append(q)

    # Initialise grid tracking states
    n_tracking_dim = 64
    tracking_lim = 1.2
    tracking_ranges = [[-tracking_lim, -tracking_lim], [tracking_lim, tracking_lim]]
    tracking_ranges = [[l, h] for l, h in zip(*tracking_ranges)]
    tracking_states = np.array(
        [list(item) for item in product(*np.array([np.linspace(l, h, n_tracking_dim) for l, h in tracking_ranges]))])
    nb_tracked = len(tracking_states)

    tracked_qvals = []
    for q in qs:
        tracked_qvals.append([])
        for action_idx in range(nb_actions):
            tracked_qvals[-1].append([])
            temp = []
            for ts in tracking_states:
                # ax.scatter(ts[0], ts[1], marker='o', c='k')
                qval = q.value(ts, action_idx)
                temp.append(qval)
            temp = np.array(temp).reshape((n_tracking_dim, n_tracking_dim))
            tracked_qvals[-1][-1].append(np.flipud(np.rot90(temp)))
    tracked_qvals = -np.squeeze(np.subtract(*tracked_qvals))



    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = np.ravel(axs)
    for action_idx, (ax, act, qvals) in enumerate(zip(axs, actions, tracked_qvals)):
        init_state = np.array([0.0, 0.0])
        env.reset(init_state.copy())
        otp1, *_ = env.step(act)
        ax.plot(*np.vstack([init_state, otp1]).T, c='k', marker='x', zorder=20)


        make_heatmap(ax, qvals, tracking_states.T[0], tracking_states.T[1], title=f'{act}')
        ax.add_patch(mpl.patches.Circle((0, 0), env.GOAL, edgecolor='g', ls='--', facecolor='None', zorder=20))

    fig.suptitle(f'{repr(env)}\n'
                 f'Training step = {actual_steps}')
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.88, wspace=0.083, hspace=0.321)
    plt.savefig(os.path.join(experiment_dir, f'change_action_qvals_from-step-{actual_steps[0]}-to-step-{actual_steps[1]}.png'))

    # fig, ax = plt.subplots()
    # qvals = []
    # for ts in tracking_states:
    #     qvals.append(np.max([q.value(ts, a_) for a_ in range(nb_actions)]))
    # qvals = np.array(qvals).reshape((n_tracking_dim, n_tracking_dim))
    # qvals = np.flipud(np.rot90(qvals))
    # make_heatmap(ax, qvals, tracking_states.T[0], tracking_states.T[1], title='Estimated values')
    # ax.add_patch(mpl.patches.Circle((0, 0), env.GOAL, edgecolor='g', ls='--', facecolor='None', zorder=20))
    # plt.savefig(os.path.join(experiment_dir, f'vals_at-step-{at_training_step}.png'))

if __name__ == '__main__':
    # plot_q_vals_grid_tracking_states()
    # plot_q_vals_region_sampling_tracking_states()

    # sub_exps = ['eps-greedy', 'boltz-1', 'boltz-5', 'boltz-10']
    exp_dir = 'changing_environment_size/exp1/sarsa_091422_214300_0/REDAClip_1.0clip_5obsx2act'
    # at_step = 80000
    # for sub_exp in sub_exps:
    #     exp_path = os.path.join(exp_dir, sub_exp)
        # plot_individual_action_advs(exp_path, at_step)
    plot_q_vals_region_sampling_tracking_states(exp_dir)
    # plot_compare_qvals_during_training()