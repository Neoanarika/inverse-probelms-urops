import torch
from einops import reduce
from torch.nn import functional as F
from src.sampling import score_fn
from pytorch_lightning import LightningModule
from utils.config import get_config_hash, get_config_base_model

def load_model(self, path="."):
    try:
        config = get_config_base_model(self.config['exp_params']['file_path'])
        hash = get_config_hash(config)
        self.load_state_dict(torch.load(f"{path}/{self.checkpoint_dir}/{hash}.ckpt"))
    except  FileNotFoundError:
        print(f"Please train the model using python run.py -c {self.config['exp_params']['file_path']}")
    return self 

class MethodNotAvaliable(Exception):
    """Exception handling for when method of a class is not avaliable"""
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload # you could add more args
    def __str__(self):
        return str(self.message) # __str__() obviously expects a string to be returned, so make sure not to send any other data types

class BaseVAE(LightningModule):

    def __init__(self, vae):
        super(BaseVAE, self).__init__()
        self.model = vae
    
    def decode(self, z):
        return self.model.decode(z)
    
    def encode(self, x):
        return self.model.det_encode(x)

    def get_random_latent(self, num):
        return self.model.model.get_random_latent(num)
    
    def to(self, device):
        self.model = self.model.to(device)
        self.model.model = self.model.model.to(device)
        return self
    
    def load_model(self, path="."):
        self.model.load_model(path=path)
    
    def forward(self):
        pass

class BaseGAN(LightningModule):

    def __init__(self, gan):
        super(BaseGAN, self).__init__()
        self.model = gan
    
    def forward(self):
        pass
    
    def decode(self, z):
        return self.model(z)
    
    def encode(self, x):
        raise MethodNotAvaliable("GANs cannot encode images into the latent space")

    def get_random_latent(self, num):
        return self.model.model.get_noise(num)
    
    def to(self, device):
        self.model = self.model.to(device)
        self.model.model = self.model.model.to(device)
        return self
    
    def load_model(self, path="."):
        self.model.load_model(path=path)

class GANModule(LightningModule):

    def __init__(
            self,
            config, 
            model
    ):
        super(GANModule, self).__init__()
        self.model = model
        self.config = config
        self.lr = config["optimizer_params"]["LR"]
        self.beta1 = config["optimizer_params"]["beta1"]
        self.dataset = config["exp_params"]["dataset"]
        self.checkpoint_dir = config["exp_params"]["checkpoint_path"]

    def forward(self, z):
        return self.model(z)
    
    def configure_optimizers(self):
        lr = self.lr
        betas = (self.beta1, 0.999)
        opt_disc = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self.model.disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self.model.gen_step(real)

        return result
    
    def load_model(self, path="."):
        self = load_model(self, path=path)

class VAEModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
            self,
            config, 
            model
    ):

        super(VAEModule, self).__init__()

        self.model = model
        self.config = config
        self.lr = config["exp_params"]["LR"]
        self.dataset = config["exp_params"]["dataset"]
        self.checkpoint_dir = config["exp_params"]["checkpoint_path"]

    def forward(self, x):
        return self.model(x)

    def det_encode(self, x):
        mu, _ = self.model.encoder(x)
        return mu

    def stoch_encode(self, x):
        mu, log_var = self.model.encoder(x)
        z = self.model.sample(mu, log_var)
        return z

    def decode(self, z):
        return self.model.decoder(z)

    def get_samples(self, num):
        return self.model.get_samples(num, self.device)

    def step(self, batch, batch_idx):
        x, y = batch

        loss = self.model.compute_loss(x)

        logs = {
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def load_model(self, path="."):
        self = load_model(self, path=path)

class EBMModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
            self,
            config, 
            base_model,
            operator,
            sampling_algo,
            **kwargs
    ):

        super(EBMModule, self).__init__()
        
        self.kwargs = kwargs
        self.config = config
        self.model = base_model
        self.operator = operator
        self.sampling_algo = sampling_algo
        self.dataset = config["exp_params"]["dataset"]
        self.checkpoint_dir = config["exp_params"]["checkpoint_path"]

    def forward(self, y):
        z = self.get_latent_estimate(y)
        imgs = self.model.decode(z)
        return imgs

    def use_gpu_for_discriminator(self):
        self.kwargs["discriminator"] = self.kwargs["discriminator"].to(self.device)
    
    def get_latent_estimate(self, y):
        def mse_potential(z):
            return F.mse_loss(self.operator(self.model.decode(z)), y, reduction='sum') + torch.norm(z, p=2)/2
        
        def discriminator_weighted_potential(z):
            discriminator = self.kwargs["discriminator"]
            score = discriminator(self.model.decode(z))
            return F.mse_loss(self.operator(self.model.decode(z)), y, reduction='sum') + torch.norm(z, p=2)/2 + score 
        
        def get_potential():
            if self.config['estimator_params']['potential'] == "mse":
                potential = mse_potential
            elif self.config['estimator_params']['potential'] == "discriminator_weighted":
                potential = discriminator_weighted_potential
            return potential 

        def get_avg_estimator(z):
            potential = get_potential()
            sampler = lambda z : self.sampling_algo(self.config, potential, z, device=self.device)
            samples = sampler(z)
            return reduce(samples, "nsamples batch latent_dim -> batch latent_dim", "mean")
        
        def get_last_estimator(z):
            potential = get_potential()
            sampler = lambda z : self.sampling_algo(self.config, potential, z, device=self.device)
            samples = sampler(z)
            return samples[-1]
        
        def get_denoise_avg_estimator(z):
            z = get_avg_estimator(z)
            potential = get_potential()
            return z - self.config["estimator_params"]["denoise_step_size"] * score_fn(potential, z)

        z = eval(f"self.get_{self.config['estimator_params']['initalisation']}_inital_latent_vector(y)")
        z = eval(f"get_{self.config['estimator_params']['mode']}_estimator(z)")
        return z

    def get_random_inital_latent_vector(self, x):
        z = self.model.get_random_latent(x.shape[0])
        return z
    
    def get_posterior_inital_latent_vector(self, x):
        z = self.model.encode(x)
        return z
    
    def get_map_estimate_inital_latent_vector(self, x):
        def potential(z):
            return F.mse_loss(self.operator(self.model.decode(z)), x, reduction='sum')
        z = self.get_random_inital_latent_vector(x)
        n = self.config["estimator_params"]["num_steps_map_initaliser"]
        step_size = self.config["estimator_params"]["step_size_map_initaliser"]
        for _ in range(n):
            grad = score_fn(potential, z)
            z = z.detach() - step_size * grad 
        return z
    
    def get_map_posterior_inital_latent_vector(self, x):
        z_hat = self.get_map_estimate_inital_latent_vector(x)
        x_hat = self.model.decode(z_hat)
        z = self.get_posterior_inital_latent_vector(x_hat)
        return z

    def get_estimates(self, y):
        z = self.get_latent_estimate(y)
        imgs = self.model.decode(z)
        return z, imgs

class AdaptiveEBM(LightningModule):

    def __init__(self, config, ebm) -> None:
        super(AdaptiveEBM, self).__init__()

        self.config = config 
        self.ebm = ebm 
    
    def estimate_variance(self, imgs):
        ex = reduce(imgs, "b c h w -> c h w", "mean")
        ex2 = reduce(imgs**2, "b c h w -> c h w", "mean")
        return ex2 - ex**2
    
    def top_k_pixel(self, varmap):
        var = reduce(varmap, "c h w -> h w", "mean")