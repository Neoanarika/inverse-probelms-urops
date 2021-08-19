# -*- coding: utf-8 -*-
# Stores VAE components that can be reused in future VAE projects

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional

def cross_entropy_with_kl(config, model, x, x_hat, z, mu, logvar):
    # This is the loss function for vae
    recons_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')

    kld_loss = model.kl_divergence(mu, logvar, "sum")

    loss = recons_loss + config["loss_params"]["kl_coeff"] * kld_loss
    return loss 

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def guassian_nll_with_kl(config, model, x, x_hat, z, mu, logvar):
    # This is the loss function for sigma VAE
    log_sigma = torch.tensor(((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log())
    log_sigma = softclip(log_sigma, -6)
    recons_loss = gaussian_nll(x_hat, log_sigma, x).sum()

    kld_loss = model.kl_divergence(mu, logvar, "sum")

    loss = recons_loss + config["kl_coeff"] * kld_loss
    return loss 

class ConvEncoder(nn.Module):
    def __init__(self, config):
        super(ConvEncoder, self).__init__()

        self.input_shape = config["exp_params"]["image_shape"]
        self.latent_dim = config["exp_params"]["latent_dim"]
        hidden_dims = config["encoder_params"]["hidden_dims"]

        modules = []

        if hidden_dims is None:
            hidden_dims = [28,64,64]

        all_channels = [self.input_shape[0]] + hidden_dims

        # encoder_conv_layers
        for i in range(len(hidden_dims)):
            modules.append(nn.Conv2d(all_channels[i], all_channels[i + 1],
                                            kernel_size=3, stride=2, padding=1))
            if not self.latent_dim == 2:
                modules.append(nn.BatchNorm2d(all_channels[i + 1]))
            modules.append(nn.LeakyReLU())
        
        self.encoder = nn.Sequential(*modules)
        self.flatten_out_size = self.flatten_enc_out_shape(self.input_shape)
        
        if self.latent_dim == 2:
            self.fc_mu = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.fc_mu = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )
        if self.latent_dim == 2:
            self.fc_var = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.fc_var = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def flatten_enc_out_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.encoder(x)
        self.shape_before_flattening = x.shape
        return int(np.prod(self.shape_before_flattening))

class ConvDecoder(nn.Module):
    def __init__(self, config):
        super(ConvDecoder, self).__init__()

        self.config = config
        self.input_shape = config["exp_params"]["image_shape"]
        self.latent_dim = config["exp_params"]["latent_dim"]
        hidden_dims = config["decoder_params"]["hidden_dims"]

        # Build Decoder
        modules = []

        if hidden_dims is None:
            hidden_dims = [64, 64,28,1]
        self.flatten_out_size = self.flatten_enc_out_shape(self.input_shape)

        if self.latent_dim == 2:
            self.decoder_input = nn.Linear(self.latent_dim, self.flatten_out_size)
        else:
            self.decoder_input = nn.Sequential(
                nn.Linear(self.latent_dim, self.flatten_out_size),
                nn.LeakyReLU(),
                nn.Dropout(0.2)
                )

        num = len(hidden_dims) -1 
        # decoder_trans_conv_layers
        for i in range(num - 1):
            modules.append(nn.UpsamplingNearest2d(scale_factor=2))
            modules.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                                        3, stride=1, padding=1))
            if not self.latent_dim == 2:
                modules.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())

        modules.append(nn.UpsamplingNearest2d(scale_factor=2))
        modules.append(nn.ConvTranspose2d(hidden_dims[num - 1], hidden_dims[num],
                                                    3, stride=1, padding=1))
        modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        z = self.decoder_input(z)
        result = z.view(z.size()[0], *self.shape_before_flattening[1:])
        result = self.decoder(result)
        return result
 
    def flatten_enc_out_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        encoder = ConvEncoder(self.config)
        x = encoder.flatten_enc_out_shape(self.input_shape)
        self.shape_before_flattening = encoder.shape_before_flattening
        return x

if __name__ == "__main__":
    decoder = ConvDecoder()
    z = torch.randn((1, 256))
    print(decoder(z))