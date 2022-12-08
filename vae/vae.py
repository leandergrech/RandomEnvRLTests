from typing import Any
import os
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAE(nn.Module):
    def __init__(self, in_dim ,latent_dim, hidden_dims=None):
        super(VAE, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            raise Exception('Must give proper hidden_dims')

        # Build Encoder
        modules = []
        layers_dim = [in_dim, *hidden_dims]
        for h1, h2 in zip(layers_dim[:-1], layers_dim[1:]):
            modules.append(
                nn.Sequential(
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                )
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        layers_dim = [latent_dim, *hidden_dims, in_dim]
        for h1, h2 in zip(layers_dim[:-1], layers_dim[1:]):
            modules.append(
                nn.Sequential(
                    nn.Linear(h1, h2),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

    def __repr__(self):
        return f'VAE_{self.in_dim}in_{self.latent_dim}z_2FC'

    def encode(self, x: Tensor):
        z = self.encoder(x)

        # Split into mu and var components of latent Gauss dist.
        mu = self.fc_mu(z)
        logvar = self.fc_var(z)

        return [mu, logvar]

    def decode(self, z: Tensor):
        x_tilde = self.decoder(z)
        return x_tilde

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        std = t.exp(t.Tensor(logvar / 2.))
        eps = t.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples: int, **kwargs):
        z = t.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        return self.forward(x)[0]

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def loss_function(self, *args: Any, **kwargs):
        """
        VAE loss function:
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        x_tilde = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]

        kld_weight = kwargs['kld_weight'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(x_tilde, x)

        kld_loss = t.mean(-0.5 * t.sum(1+logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss * kld_weight

        return dict(loss=loss, Reconstruction_loss=recons_loss.detach(), KLD=-kld_loss.detach())


if __name__ == '__main__':
    from utils.training_utils import TrajBuffer
    from datetime import datetime as dt

    from random_env.envs import RandomEnv

    plot_results = True

    buffer = TrajBuffer()
    par_dir = 'random_trajectory_training'

    env_szs = np.arange(2, 6)
    latent_dims = np.arange(2, 6)
    hidden_dims = [10, 10]

    nb_eps = 1000
    max_steps = 100

    batch_size = 4
    nb_training_steps = 10000
    learning_rate = 1e-3
    kld_weight = 0.01

    loss_ratio_success = 0.01

    seed = 123
    np.random.seed(seed)

    for env_sz in env_szs:
        n_obs, n_act = [env_sz] * 2
        env = RandomEnv(n_obs=env_sz, n_act=env_sz)
        env.EPISODE_LENGTH_LIMIT = max_steps

        print(f'-> {repr(env)}')

        for latent_sz in latent_dims:
            if latent_sz > env_sz:
                continue

            exp_name = f"VAE-RE_{dt.now().strftime('%m%d%y_%H%M%S')}_{env_sz}obsx{env_sz}act_{latent_sz}z_{seed}seed"
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
                f.write(f'-> Train env max steps: {max_steps}\n')
                f.write(f'-> Number of training its: {it}\n')
                f.write(f'-> Lowest loss: {min(losses)}\n')
                f.write(f'-> Loss improvement: {((losses[-1] - losses[0]) / losses[0]) * 100.0}%\n')
                f.write(f'-> Loss improvement (ptp): {((np.min([-1]) - np.max(losses)) / np.max(losses[0])) * 100.0}%\n')
                f.write(f'-> Experiment_path: {os.path.abspath(exp_path)}\n')

            if plot_results:
                plot_path = os.path.join(exp_path, 'losses.png')
                print(f' `-> Plotting losses to: {plot_path}')
                import matplotlib.pyplot as plt
                from utils.plotting_utils import grid_on
                fig, ax = plt.subplots(figsize=(12,7))
                ax.plot(losses)
                ax.axhline(0.0, ls='--', color='k')
                grid_on(ax=ax, axis='x')
                grid_on(ax=ax, axis='y')
                ax.set_xlabel('Training iterations')
                ax.set_ylabel('Loss')
                fig.tight_layout()
                fig.savefig(plot_path)


