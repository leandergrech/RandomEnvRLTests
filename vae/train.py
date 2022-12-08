import os
import numpy as np
from datetime import datetime as dt
import torch as t
import matplotlib.pyplot as plt

from utils.training_utils import TrajBuffer, get_random_alpha_numeral
from utils.plotting_utils import grid_on
from vae import VAE
from random_env.envs import RandomEnv

plot_results = True

buffer = TrajBuffer()

env_szs = np.arange(6, 11)
latent_dims = np.arange(2, 11)
hidden_dims = [32, 32]

nb_eps = 1000
max_steps = 100

batch_size = 32
nb_training_steps = 10000
learning_rate = 1e-3
kld_weight = 0.01

loss_ratio_success = 0.01

seed = 123

par_dir = f'random_trajectory_{batch_size}batch_{learning_rate}lr_{kld_weight}klw_{seed}seed'

for env_sz in env_szs:
    np.random.seed(seed)

    n_obs, n_act = [env_sz] * 2
    env = RandomEnv(n_obs=env_sz, n_act=env_sz) # Square dynamics
    env.EPISODE_LENGTH_LIMIT = max_steps

    print(f'-> Dynamics of {repr(env)}')

    for latent_sz in latent_dims:
        if latent_sz > env_sz:
            continue

        # exp_name = f"VAE-RE_{dt.now().strftime('%m%d%y_%H%M%S')}_{env_sz}obsx{env_sz}act_{latent_sz}z_{seed}seed"
        exp_name = f"VAE-RE_{get_random_alpha_numeral()}_{env_sz}obsx{env_sz}act_{latent_sz}z_{seed}seed"
        exp_path = os.path.join(par_dir, exp_name)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        else:
            raise FileExistsError

        vae = VAE(in_dim=n_obs, latent_dim=latent_sz, hidden_dims=hidden_dims)

        optim = t.optim.SGD(vae.parameters(), lr=learning_rate)

        print(f' `-> Created {repr(vae)}')
        print(f'   `-> Collecting episodes')

        for ep in range(nb_eps):
            buffer.reset()
            o = env.reset()
            d = False
            while not d:
                a = env.action_space.sample()
                otp1, r, d, info = env.step(a)

                buffer.add(o, a, r, otp1)
                o = otp1.copy()

        print(f'   `-> Training VAE')
        losses = []
        it = 0
        for it in range(nb_training_steps):
            o, a, r, otp1 = buffer.sample_batch(batch_size=batch_size)

            optim.zero_grad()

            o_tilde, _, mu, logvar = vae.forward(t.Tensor(o))
            o = t.Tensor(o)
            loss = vae.loss_function(o_tilde, o, mu, logvar, kld_weight=kld_weight)['loss']

            loss.backward()
            optim.step()

            loss = loss.item()
            losses.append(loss)
            # Stop training early
            if loss < losses[0] * loss_ratio_success:
                break

        print(f'   `-> Saving to: {exp_path}')

        t.save(vae, os.path.join(exp_path, f"{repr(vae)}.pkl"))
        env.save_dynamics(exp_path)
        np.save(os.path.join(exp_path, 'losses.npy'), np.array(losses))
        with open(os.path.join(exp_path, 'info.txt'), 'w') as f:
            f.write(f'-> Experiment name: {exp_name}\n')
            f.write(f'-> Environment name: {repr(env)}\n')
            f.write(f'-> Latent size: {latent_sz}\n')
            f.write(f'-> learning_rate: {learning_rate}\n')
            f.write(f'-> hidden_dims: {hidden_dims}\n')
            f.write(f'-> batch_size: {batch_size}\n')
            f.write(f'-> kld_weight: {kld_weight}\n')
            f.write(f'-> loss_ratio_success: {loss_ratio_success}\n')
            f.write(f'-> Train env max steps: {max_steps}\n')
            f.write(f'-> Number of training its: {it + 1}\n')
            f.write(f'-> Lowest loss @ step: {min(losses)}@{np.argmin(losses)}\n')
            f.write(f'-> Highest loss @ step: {max(losses)}@{np.argmax(losses)}\n')
            f.write(f'-> Loss improvement: {((losses[-1] - losses[0]) / losses[0]) * 100.0}%\n')
            f.write(f'-> Loss improvement (ptp): {((np.min([-1]) - np.max(losses)) / np.max(losses[0])) * 100.0}%\n')
            f.write(f'-> Experiment_path: {os.path.abspath(exp_path)}\n')

        if plot_results:
            plot_path = os.path.join(exp_path, 'losses.png')
            print(f' `-> Plotting losses to: {plot_path}')

            fig, ax = plt.subplots(figsize=(12,7))
            ax.plot(losses)
            ax.axhline(0.0, ls='--', color='k')
            grid_on(ax=ax, axis='x')
            grid_on(ax=ax, axis='y')
            ax.set_xlabel('Training iterations')
            ax.set_ylabel('Loss')
            fig.tight_layout()
            fig.savefig(plot_path)
