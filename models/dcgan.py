import torch
from torch import nn, Tensor
from typing import Any, Callable
from pytorch_lightning import LightningModule

class GAN(LightningModule):
    """
    DCGAN implementation.
    Example::
        from pl_bolts.models.gans import DCGAN
        m = DCGAN()
        Trainer(gpus=2).fit(m)
    Example CLI::
        # mnist
        python dcgan_module.py --gpus 1
        # cifar10
        python dcgan_module.py --gpus 1 --dataset cifar10 --image_channels 3
    """

    def __init__(
        self,
        name: str,
        generator: Callable,
        discriminator: Callable
    ) -> None:
        """
        Args:
            beta1: Beta1 value for Adam optimizer
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            image_channels: Number of channels of the images from the dataset
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate
        """
        super().__init__()
        
        self.name = name
        self.criterion = nn.BCELoss()
        self.latent_dim = generator.latent_dim
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, noise: Tensor) -> Tensor:
        """
        Generates an image given input noise
        Example::
            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def disc_step(self, real: Tensor) -> Tensor:
        disc_loss = self.get_disc_loss(real)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def gen_step(self, real: Tensor) -> Tensor:
        gen_loss = self.get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def get_disc_loss(self, real: Tensor) -> Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self.get_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def get_gen_loss(self, real: Tensor) -> Tensor:
        # Train with fake
        fake_pred = self.get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self.criterion(fake_pred, fake_gt)

        return gen_loss

    def get_samples(self, num):
        noise = self.get_noise(num)
        fake = self(noise)
        return fake
        
    def get_fake_pred(self, real: Tensor) -> Tensor:
        batch_size = len(real)
        noise = self.get_noise(batch_size)
        fake = self(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def get_noise(self, n_samples: int) -> Tensor:
        return torch.randn(n_samples, self.latent_dim, device=self.device)