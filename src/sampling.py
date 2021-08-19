import torch
import numpy as np

# Basically there is a standard interface to these functions and we should check that all of tehm follows this
# Write testing code to ensure the shape is a 
# 3-dim shape corresponding to n_samples, batch, num
# This can help us debug future sampling algorithm

def score_fn(potential, Zi):
    Zi.requires_grad_()
    u = potential(Zi).mean()
    grad = torch.autograd.grad(u, Zi)[0]
    return grad 

def map(config, potential, Zi):
    n = config["estimator_params"]["n"]
    step_size = config["estimator_params"]["step_size"]
    samples = torch.zeros((1, ) + Zi.shape)

    for idx in range(n):
        grad = score_fn(potential, Zi)
        Zi = Zi.detach() - step_size * grad 
    samples[0] = Zi
    return samples 

def langevin(config, potential, Zi):
    burn_in = config["estimator_params"]["burn_in"]
    n_samples = config["estimator_params"]["n_samples"]
    step_size = config["estimator_params"]["step_size"]
    samples = torch.zeros((n_samples, ) + Zi.shape)

    for idx in range(n_samples + burn_in):
        grad = score_fn(potential, Zi)
        Zi = Zi.detach() - step_size * grad  + \
        np.sqrt(2 * step_size) * torch.randn_like(Zi)
        if idx > burn_in:
            samples[idx-burn_in] = Zi
    return samples 

def log_Q(z_prime, z, potential, step):
    z.requires_grad_()
    grad = torch.autograd.grad(potential(z).mean(), z)[0]
    return -(torch.norm(z_prime - z + step * grad, p=2, dim=1) ** 2) / (4 * step)

def mala(config, potential, Zi):
    burn_in = config["estimator_params"]["burn_in"]
    n_samples = config["estimator_params"]["n_samples"]
    step = config["estimator_params"]["step_size"]
    samples = torch.zeros((n_samples, ) + Zi.shape)

    for idx in range(n_samples + burn_in):
        grad = score_fn(potential, Zi)
        Znew = Zi.detach() - step * grad + np.sqrt(2 * step) * torch.randn(1, 2) 
        u = np.random.uniform()
        alpha = min(1, torch.exp(potential(Zi-potential(Znew)\
                                 + log_Q(Zi, Znew, potential, step) - log_Q(Znew, Zi, potential, step))))
        if u< alpha:
          Zi = Znew
        if idx > burn_in:
            samples[idx-burn_in] = Zi
    return samples 

def hmc(config, potential, Zi):
    burn_in = config["estimator_params"]["burn_in"]
    L = config["estimator_params"]["L"]
    n_samples = config["estimator_params"]["n_samples"]
    step = config["estimator_params"]["step_size"]
    samples = torch.zeros((n_samples, ) + Zi.shape)

    for idx in range(n_samples + burn_in):
        grad = score_fn(potential, Zi)
        velocity = torch.randn(1, 2)
        velocity_new = velocity - 0.5 * step*grad 
        Znew = Zi.detach() + step*velocity_new 
        for _ in range(L):
          Zi.requires_grad_()
          u = potential(Zi).mean()
          grad = torch.autograd.grad(u, Zi)[0]
          velocity_new = velocity_new -step* grad 
          Znew = Znew.detach() + step*velocity_new
        
        u = potential(Zi).mean()
        grad = torch.autograd.grad(u, Zi)[0]
        velocity_new = velocity - 0.5 * step*grad 
        u = np.random.uniform()
        alpha = min(1, torch.exp(potential(Zi)+(velocity).mm(velocity.t())-potential(Znew)-(velocity_new).mm(velocity_new.t())))
        if u< alpha:
          Zi = Znew
          velocity=velocity_new
        if idx > burn_in:
            samples[idx-burn_in] = Zi
    
    return samples 

def annealed_langevin_algorithm(config, potential, Zi):
    burn_in = config["estimator_params"]["burn_in"]
    L = config["estimator_params"]["L"]
    T = config["estimator_params"]["T"]
    var = config["estimator_params"]["varience_levels"]
    eps = config["estimator_params"]["step_size"]

    idx = 0
    samples = torch.zeros((L*T-burn_in, ) + Zi.shape)

    for i in range(L):
      alpha_i  = eps*var[i]/var[-1]
      for _ in range(T):
        grad = score_fn(potential, Zi)
        Zi = Zi.detach() - (alpha_i/2) * grad + np.sqrt(alpha_i) * torch.randn(1, 2)
        if idx > burn_in:
            samples[idx-burn_in] = Zi
        idx+=1

    return samples