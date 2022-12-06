from typing import Any
import os
import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAE(nn.Module):
    def __init__(self, in_dim ,latent_dim, hidden_dims = None):
        super(VAE, self).__init__()
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
        return [self.decoder(z), x, mu, logvar]

    def loss_function(self, *args: Any, **kwargs):
        """
        VAE loss function:
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        x_tilde = args[0]
        x = args[1]
        mu = args[2]
        logvar = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(x_tilde, x)

        kld_loss = t.mean(-0.5 * t.sum(1+logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss * kld_weight

        return dict(loss=loss, Reconstruction_loss=recons_loss.detach(), KLD=-kld_loss.detach())

