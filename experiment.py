import torch
from torch import nn, Tensor
from einops import reduce
from abc import ABC, abstractmethod
from torch.nn import functional as F
from src.sampling import score_fn, map
from pytorch_lightning import LightningModule
from utils.config import get_config_hash, get_config_base_model

def load_model(self):
    try:
        config = get_config_base_model(self.config['exp_params']['file_path'])
        hash = get_config_hash(config)
        self.load_state_dict(torch.load(f"./{self.checkpoint_dir}/{hash}.ckpt"))
    except  FileNotFoundError:
        print(f"Please train the model using python run.py -c {self.config['exp_params']['file_path']}")
    return self 

class MethodNotAvaliable(Exception):
    """Still an exception raised when uncommon things happen"""
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload # you could add more args
    def __str__(self):
        return str(self.message) # __str__() obviously expects a string to be returned, so make sure not to send any other data types

class BaseModel(ABC):

    @abstractmethod
    def to(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass
    
    @abstractmethod
    def get_random_latent(self):
        pass

class BaseVAE(BaseModel):

    def __init__(self, vae):
        self.model = vae
    
    def decode(self, z):
        return self.model.decode(z)
    
    def encode(self, x):
        return self.model.det_encode(x)

    def get_random_latent(self, num):
        return self.model.model.get_random_latent(num)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def load_model(self):
        self.model.load_model()

class BaseGAN(BaseModel):

    def __init__(self, gan):
        self.model = gan
    
    def decode(self, z):
        return self.model(z)
    
    def encode(self, x):
        raise MethodNotAvaliable("GANs cannot encode images into the latent space")

    def get_random_latent(self, num):
        return self.model.model.get_noise(num)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def load_model(self):
        self.model.load_model()

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
        z = z.to(self.device)
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
    
    def load_model(self):
        self = load_model(self)

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
        x = x.to(self.device)
        return self.model(x)

    def det_encode(self, x):
        x = x.to(self.device)
        mu, _ = self.model.encoder(x)
        return mu

    def stoch_encode(self, x):
        x = x.to(self.device)
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

    def load_model(self):
        self = load_model(self)

class EBMModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
            self,
            config, 
            base_model,
            operator,
            sampling_algo
    ):

        super(EBMModule, self).__init__()
        
        self.config = config
        self.model = base_model
        self.operator = operator
        self.sampling_algo = sampling_algo
        self.dataset = config["exp_params"]["dataset"]
        self.checkpoint_dir = config["exp_params"]["checkpoint_path"]

    def forward(self, y):
        y = y.to(self.device)
        self.model = self.model.to(self.device)
        self.operator = self.operator.to(self.device)
        z = self.get_latent_estimate(y)
        imgs = self.model.decode(z)
        return imgs

    def get_latent_estimate(self, y):
        def potential(z):
            return F.mse_loss(self.operator(self.model.decode(z)), y, reduction='sum')
        
        def get_avg_estimator(z):
            sampler = lambda z : self.sampling_algo(self.config, potential, z)
            samples = sampler(z)
            return reduce(samples, "nsamples batch latent_dim -> batch latent_dim", "mean")
        
        def get_last_estimator(z):
            sampler = lambda z : self.sampling_algo(self.config, potential, z)
            samples = sampler(z)
            return samples[-1]
        
        def get_denoise_avg_estimator(z):
            z = get_avg_estimator(z)
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
        y = y.to(self.device)
        self.model = self.model.to(self.device)
        self.operator = self.operator.to(self.device)
        z = self.get_latent_estimate(y)
        imgs = self.model.decode(z)
        return z, imgs