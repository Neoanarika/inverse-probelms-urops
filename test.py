import torch 
import unittest
from models.mnist.gan import DCGANGenerator, DCGANDiscriminator
from utils.config import get_config_base_model

# Unittest
class TestGAN(unittest.TestCase):

    def test_dcgan_generator(self):
        config = get_config_base_model("./configs/mnist/gan/dcgan.yaml")
        gen = DCGANGenerator(config)
        noise = torch.randn(1, config["exp_params"]["latent_dim"])
        noise = noise.view(*noise.shape, 1, 1)
        assert gen(noise).shape == (1,  1, 32, 32)

    def test_dcgan_discriminator(self):
        config = get_config_base_model("./configs/mnist/gan/dcgan.yaml")
        disc = DCGANDiscriminator(config)
        img = torch.randn(1, 1, 32, 32)
        disc(img)

if __name__ == "__main__":
    unittest.main()