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

    buffer = TrajBuffer()
    par_dir = 'random_trajectory_training'

    env_szs = np.arange(2, 6)
    latent_dims = np.arange(2, 6)
    hidden_dims = [10, 10]

    nb_eps = 3
    max_steps = 10

    batch_size = 4
    nb_training_steps = 100
    learning_rate = 1e-2

    seed = 123
    np.random.seed(seed)

    for env_sz in env_szs:
        n_obs, n_act = [env_sz] * 2
        env = RandomEnv(n_obs=env_sz, n_act=env_sz)
        env.EPISODE_LENGTH_LIMIT = max_steps

        print(f'-> {repr(env)}')

        for latent_sz in latent_dims:
            vae = VAE(in_dim=n_obs, latent_dim=latent_sz, hidden_dims=hidden_dims)

            optim = t.optim.SGD(vae.parameters(), lr=learning_rate)

            print(f' `-> {repr(vae)}')
            print(f' `-> Collecting episodes')

            for ep in range(nb_eps):
                buffer.reset()
                o = env.reset()
                d = False
                while not d:
                    a = env.action_space.sample()
                    otp1, r, d, info = env.step(a)

                    buffer.add(o, a, r, otp1)
                    o = otp1.copy()

            print(f' `-> Training VAE')
            losses = []
            for it in range(nb_training_steps):
                o, a, r, otp1 = buffer.sample_batch(batch_size=batch_size)

                optim.zero_grad()

                o_tilde, _, mu, logvar = vae.forward(t.Tensor(o))
                loss = vae.loss_function(o_tilde, o, mu, logvar, kld_weight=0.01)['loss']

                loss.backward()
                optim.step()

                losses.append(loss.item())

            project_name = f"{dt.now().strftime('%m%d%y_%H%M%S')}_{env_sz}obsx{env_sz}act_{latent_sz}z_{seed}seed"
            vae_name = f"{repr(vae)}.pkl"
            t.save(vae, os.path.join(par_dir, project_name, vae_name))
            env.save_dynamics(os.path.join(par_dir, project_name))
            np.save(os.path.join(par_dir, project_name, 'losses.npy'), np.array(losses))



