import torch
from torch import nn, Tensor
from utils.config import get_config_base_model
import torch.nn.functional as F

class DCGANGenerator(nn.Module):

    def __init__(self, config) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        latent_dim = config["exp_params"]["latent_dim"]
        feature_maps = config["generator_params"]["feature_maps"]
        image_channels = config["generator_params"]["image_channels"]
        
        self.latent_dim = latent_dim
        self.gen = nn.Sequential(
            self._make_gen_block(latent_dim, feature_maps * 8, kernel_size=2, stride=1, padding=0),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )

    @staticmethod
    def _make_gen_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Mish(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return gen_block

    def forward(self, noise: Tensor) -> Tensor:
        return self.gen(noise)


class DCGANDiscriminator(nn.Module):

    def __init__(self, config) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        latent_dim = config["exp_params"]["latent_dim"]
        feature_maps = config["discriminator_params"]["feature_maps"]
        image_channels = config["discriminator_params"]["image_channels"]

        self.latent_dim = latent_dim
        self.disc = nn.Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(feature_maps * 8, 1, kernel_size=2, stride=1, padding=0, last_block=True),
        )

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        batch_norm: bool = True,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            )

        return disc_block

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(self.disc(x)).view(-1, 1).squeeze(1)
    
    def logit(self, x: Tensor) -> Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)

if __name__ == "__main__":
    config = get_config_base_model("./configs/mnist/gan/dcgan.yaml")
    gen = DCGANGenerator(config)
    noise = torch.randn(1, config["exp_params"]["latent_dim"])
    print(gen(noise))