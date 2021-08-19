import torch
from typing import Callable
from pytorch_lightning import LightningModule

class VAE(LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
        self,
        name: str,
        loss: Callable,
        encoder: Callable,
        decoder: Callable,
        **kwargs
    ) -> None:

        super(VAE, self).__init__()

        self.name = name
        self.loss = loss
        self.kwargs = kwargs
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = self.encoder.latent_dim

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        return z, self.decoder(z), mu, log_var
    
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_loss(self, x):
        z, x_hat, mu, logvar = self._run_step(x)
        loss = self.loss(self, x, x_hat, z , mu, logvar)

        return loss
    
    def get_random_latent(self, num):
        z = torch.randn(num, self.encoder.latent_dim, device=self.device)
        return z

    def get_samples(self, num):
        z = self.get_random_latent(num)
        return self.decoder(z)

    def kl_divergence(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - torch.square(mean) - torch.exp(logvar))