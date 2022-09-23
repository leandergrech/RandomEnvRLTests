import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
from datetime import datetime as dt
from tqdm import tqdm

from stable_baselines3 import PPO
from random_env.envs import RandomEnv

par_dir = 'param_random_env_results'

N_STEPS = 300
DPI = 50
ANIMATION_INTERVAL = 50
NB_EVALUATION_GIFS = 3


# Evaluate with a scaled initial state to test `easier` initial conditions closer to the optimal state
# INIT_STATE_SCALE = 1.0


def get_action_ylims(action):
    discrete_val = 0.5

    mina, maxa = abs(min(action)), abs(max(action))

    ylim_min = -np.ceil(mina / discrete_val) * discrete_val
    ylim_max = np.ceil(maxa / discrete_val) * discrete_val

    ylim = max(np.abs([ylim_min, ylim_max]))

    return (-ylim, ylim)


def save_agent_vs_optimal_gif(model, env, opt_env, save_path, title_name, init_state=None):
    n_obs = env.obs_dimension
    n_act = env.act_dimension

    o_x = range(n_obs)
    a_x = range(n_act)

    state_t0 = np.zeros(n_obs)
    init_action = np.zeros(n_act)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title_name)
    gs = fig.add_gridspec(3, 2)

    ax_state = fig.add_subplot(gs[0, 0])
    ax_action = fig.add_subplot(gs[1, 0])
    ax_state_opt = fig.add_subplot(gs[0, 1])
    ax_action_opt = fig.add_subplot(gs[1, 1])
    ax_rewards = fig.add_subplot(gs[2, :])

    plt.subplots_adjust(left=0.09, bottom=0.08, right=.95, top=.87, wspace=0.2, hspace=0.2)

    for i, ax in enumerate(fig.axes):
        if i != len(fig.axes) - 1:
            ax.grid(axis='y')

    # state axes
    temp = [None] * 2
    for i, (ax, title) in enumerate(zip((ax_state, ax_state_opt), ('State - Agent', 'State - Optimal'))):
        temp[i] = ax.bar(o_x, state_t0)
        ax.axhline(0.0, color='k', ls='dashed')
        ax.axhline(env.GOAL, color='g', ls='dashed', label='Goal')
        ax.axhline(-env.GOAL, color='g', ls='dashed')
        ax.set_title(title)
        ax.set_ylim((-1, 1))
    o_bars, o_bars_opt = temp

    # action axes
    temp = [None] * 2
    for i, (ax, title) in enumerate(zip((ax_action, ax_action_opt), ('Action - Agent', 'Action - Optimal'))):
        temp[i] = ax.bar(a_x, init_action)
        ax.set_title(title)
        ax.set_ylim(get_action_ylims(init_action))
    a_bars, a_bars_opt = temp

    # ax_rewards
    rew_line, = ax_rewards.plot([], [], label='Rewards - Agent')
    rew_line_opt, = ax_rewards.plot([], [], label='Rewards - Optimal')
    ax_rewards.axhline(abs(env.reward_thresh), color='g', ls='dashed', label='Reward threshold')
    ax_rewards.set_xlabel('Steps')
    ax_rewards.set_ylabel('|Reward|')
    ax_rewards.set_yscale('log')
    ax_rewards.grid(which='both')
    rew_ylim = [-0.1, 1]

    for i, ax in enumerate(fig.axes):
        ax.legend(loc='upper right')

    o_bars.remove()
    a_bars.remove()
    o_bars_opt.remove()
    a_bars_opt.remove()

    o, opt_o = np.zeros(env.obs_dimension), np.zeros(env.obs_dimension)
    a, opt_a = init_action, init_action

    o_bars = ax_state.bar(o_x, o, color='b')
    a_bars = ax_action.bar(a_x, a, color='r')
    o_bars_opt = ax_state_opt.bar(o_x, opt_o, color='b')
    a_bars_opt = ax_action_opt.bar(a_x, opt_a, color='r')

    def update_bars(o, a, opo, opa):
        nonlocal o_bars, a_bars, o_bars_opt, a_bars_opt, o_x, a_x
        nonlocal ax_state, ax_action, ax_state_opt, ax_action_opt

        # for bar in (o_bars, a_bars, o_bars_opt, a_bars_opt):
        #     bar.remove()

        for bar, dat in zip((o_bars, a_bars, o_bars_opt, a_bars_opt), (o, a, opo, opa)):
            for b, d in zip(bar, dat):
                b.set_height(d)

    rewards = []
    rewards_opt = []

    # o = env.reset()
    # opt_o = o.copy()
    # opt_env.reset(opt_o)

    d_agent, d_opt = [False] * 2

    def animate(i):
        nonlocal o_bars, a_bars, o_bars_opt, a_bars_opt
        nonlocal ax_state, ax_state_opt
        nonlocal a, o, opt_a, opt_o, d_agent, d_opt
        nonlocal rewards, rewards_opt, rew_ylim

        ep_idx = int(i / N_STEPS)
        step_idx = i % N_STEPS

        # Environments traversal logic
        if (d_agent and d_opt) or i == 0:
            d_agent, d_opt = [False] * 2
            if init_state is None:
                o = env.reset(env.observation_space.sample())
            else:
                o = np.copy(init_state)
                env.reset(init_state=o)
            opt_o = np.copy(init_state)
            opt_env.reset(opt_o)

            a, opt_a = init_action, init_action
            update_bars(o, a, opt_o, opt_a)

            rewards.append(-env.objective(o))
            rewards_opt.append(-env.objective(opt_o))

        if not d_agent:
            a = model.predict(o, deterministic=True)[0]
            o, r, d_agent, _ = env.step(a)
            rewards.append(-r)

        if not d_opt:
            opt_a = opt_env.get_optimal_action(opt_o)
            opt_o, opt_r, d_opt, _ = opt_env.step(opt_a)
            rewards_opt.append(-opt_r)

        # Update reward plot
        rew_line.set_data(range(len(rewards)), rewards)
        rew_line_opt.set_data(range(len(rewards_opt)), rewards_opt)
        ax_rewards.set_xlim((-2, max(len(rewards), len(rewards_opt)) + 2))
        max_reward = max(np.concatenate([rewards, rewards_opt, [rew_ylim[1]]]))
        ax_rewards.set_ylim((rew_ylim[0], 1.1 * max_reward))
        rew_ylim = ax_rewards.get_ylim()

        # Update state and action plots
        ax_state.set_title(f'Ep #{ep_idx} - Step #{step_idx} - Done {d_agent}\nState - Agent')
        ax_state_opt.set_title(f'Ep #{ep_idx} - Step #{step_idx} - Done {d_opt}\nState - Optimal')
        update_bars(o, a, opt_o, opt_a)
        for ax, dat in zip((ax_state, ax_state_opt), (o, opt_o)):
            ax.set_ylim((min(-1, min(dat)), max(1, max(dat))))
        ax_action.set_ylim(get_action_ylims(a))
        ax_action_opt.set_ylim(get_action_ylims(a))

    # return o_bars, a_bars, o_bars_opt, a_bars_opt, fig.axes

    print('Starting animation')
    anim = FuncAnimation(fig=fig, func=animate, frames=N_STEPS,
                         interval=ANIMATION_INTERVAL, blit=False, repeat=False)

    # plt.show()
    print(f'Saving animation to: {save_path}')
    anim.save(save_path, dpi=DPI)
    print('Saved to: ', save_path)


def setup_experiment(load_dir, model_name, training_step):
    env, opt_env, agent = [None] * 3
    for sub_dir in os.listdir(load_dir):
        if model_name not in sub_dir:
            continue
        env = RandomEnv.load_from_dir(os.path.join(load_dir, sub_dir))
        opt_env = RandomEnv(env.obs_dimension, env.act_dimension, model_info=env.model_info)

        agent_name = f'{model_name}_{training_step}_steps.zip'
        agent = PPO.load(os.path.join(load_dir, sub_dir, agent_name), env)

    assert agent is not None, f'Cannot find `{model_name}` in directory: {load_dir}'

    return env, opt_env, agent


def gif_of_one_agent_at_specific_training_step(training_session_date, training_seed, training_step, init_state=None):
    global par_dir

    load_dir = os.path.join(par_dir, f'training_session_pt_{training_session_date}')
    eval_dir = os.path.join(load_dir, 'evals')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    n_obs, n_act = 5, 5

    if training_seed is None:
        model_name = f'PPO_RandomEnv_{n_obs}obsx{n_act}act'
    else:
        model_name = f'PPO_RandomEnv_{n_obs}obsx{n_act}act_seed{training_seed}'

    # Load environment dyanamics used to train the agent and the trained agent at the requested training step
    env, opt_env, agent = setup_experiment(load_dir, model_name, training_step)

    # GIFs creation loop
    # for _ in tqdm(range(NB_EVALUATION_GIFS)):
    save_path = os.path.join(eval_dir, f'{model_name}_{training_step}_steps_{dt.strftime(dt.now(), "%H%M%S")}')
    if INIT_STATE_SCALE < 1:
        save_path += f'_initstatescale{INIT_STATE_SCALE}'
    save_path += '.gif'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_agent_vs_optimal_gif(agent, env, opt_env,
                                  save_path,
                                  model_name + f' @ {training_step} training steps',
                                  init_state=init_state)


def gifs_of_agent_performance_during_training_same_init_state():
    global par_dir, INIT_STATE_SCALE, NB_EVALUATION_GIFS

    training_session_date = '021122_013057'
    load_dir = os.path.join(par_dir, f'training_session_pt_{training_session_date}')
    eval_dir = os.path.join(load_dir, 'evals')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    EVALUATE_SEED = 123
    for EVALUATE_SEED in (123, 234, 345, 456, 567):
        EVALUATE_STEP_START = 10000
        EVALUATE_STEP_END = 10001
        EVALUATE_STEP_INT = 10000

        required_steps = np.arange(EVALUATE_STEP_START, EVALUATE_STEP_END + 1, EVALUATE_STEP_INT)

        n_obs, n_act = 5, 5

        if EVALUATE_SEED is None:
            model_name = f'PPO_RandomEnv_{n_obs}obsx{n_act}act'
        else:
            model_name = f'PPO_RandomEnv_{n_obs}obsx{n_act}act_seed{EVALUATE_SEED}'

        available_steps = []
        for sub_model_file in os.listdir(os.path.join(load_dir, model_name)):
            if model_name not in sub_model_file:
                continue
            else:
                available_steps.append(int(sub_model_file.split('_')[-2]))

        available_steps = sorted(available_steps)
        for INIT_STATE_SCALE in (1.0, 0.5, 0.2):

            init_states = np.random.uniform(-1, 1, size=NB_EVALUATION_GIFS * n_obs).reshape((NB_EVALUATION_GIFS, n_obs)) \
                          * INIT_STATE_SCALE

            for EVAL_STEP in available_steps:
                if EVAL_STEP in required_steps:
                    for i in range(NB_EVALUATION_GIFS):
                        print(f'-> {model_name} - INIT_STATE_SCALE={INIT_STATE_SCALE} - GIF #{i + 1}')
                        gif_of_one_agent_at_specific_training_step(training_session_date=training_session_date,
                                                                   training_seed=EVALUATE_SEED,
                                                                   training_step=EVAL_STEP,
                                                                   init_state=np.copy(init_states[i]))


if __name__ == '__main__':
    gifs_of_agent_performance_during_training_same_init_state()
