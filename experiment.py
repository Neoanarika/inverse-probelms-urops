import torch
from einops import reduce
from abc import ABC, abstractmethod
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from utils.config import get_config_hash, get_config_base_model

class BaseModel(ABC):

    @abstractmethod
    def to(self):
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

class BaseGAN(BaseModel):

    def __init__(self, gan):
        self.model = gan
    
    def decode(self, z):
        return self.model(z)
    
    def encode(self, x):
        raise "GANs cannot encode images into the latent space"

    def get_random_latent(self, num):
        return self.model.model.get_noise(num)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

class GANModule(LightningModule):

    def __init__(
            self,
            config, 
            model
    ):
        self.model = model
        self.config = config
        self.lr = config["exp_params"]["LR"]
        self.beta1 = config["exp_params"]["beta1"]
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
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

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
        try:
            config = get_config_base_model(self.config['exp_params']['file_path'])
            hash = get_config_hash(config)
            self.load_state_dict(torch.load(f"./{self.checkpoint_dir}/{hash}.ckpt"))
        except  FileNotFoundError:
            print(f"Please train the model using python run.py -c {self.config['exp_params']['file_path']}")

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
        
        def get_avg_estimate(z):
            sampler = lambda z : self.sampling_algo(self.config, potential, z)
            samples = sampler(z)
            return reduce(samples, "nsamples batch latent_dim -> batch latent_dim", "mean")
        
        def get_last_estimate(z):
            sampler = lambda z : self.sampling_algo(self.config, potential, z)
            samples = sampler(z)
            return samples[-1]
        
        def get_denoise_avg_estimate(z):
            z = get_avg_estimate(z)


        z = self.get_posterior_inital_latent_vector(y)
        z = get_avg_estimate(z)
        return z

    def get_inital_latent_vector(self, x):
        z = self.model.get_random_latent(x.shape[0])
        return z
    
    def get_posterior_inital_latent_vector(self, x):
        z = self.model.encode(x)
        return z
    
    def get_estimates(self, y):
        y = y.to(self.device)
        self.model = self.model.to(self.device)
        self.operator = self.operator.to(self.device)
        z = self.get_latent_estimate(y)
        imgs = self.model.decode(z)
        return z, imgs