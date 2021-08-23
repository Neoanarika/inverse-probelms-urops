import torch 
import unittest
from torch import nn
from models.mnist.gan import DCGANGenerator, DCGANDiscriminator
from utils.config import get_config_base_model
from assembler import make_energy_model, get_config, get_config_ebm

def get_model_config(model_name):
    dataset, model, sampling, task = model_name.split("/")
    name = f"{sampling}/{task}"
    config = get_config(get_config_ebm, dataset, model, name)
    return config

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

class TestOperators(unittest.TestCase):

    def test_random_occlude(self):
        model_name = "mnist/vae/langevin/inpainting"
        config = get_model_config(model_name)
        config["operator_params"]["operator"] = "RandomOcclude"
        config["operator_params"]["num_measurements"] = 200
        ebm = make_energy_model(config)
        x = torch.randn(1, 1, 32, 32)
        x_tilde = ebm.operator(x)

        assert x_tilde.shape == (1,  1, 32, 32)

    def test_center_occlude(self):
        model_name = "mnist/vae/langevin/inpainting"
        config = get_model_config(model_name)
        config["operator_params"]["operator"] = "CenterOcclude"
        config["operator_params"]["size"] = 13
        ebm = make_energy_model(config)
        x = torch.randn(1, 1, 32, 32)
        x_tilde = ebm.operator(x)

        assert x_tilde.shape == (1,  1, 32, 32)
    
    def test_compressed_sensing(self):
        model_name = "mnist/vae/langevin/inpainting"
        config = get_model_config(model_name)
        config["operator_params"]["operator"] = "CompressedSensing"
        config["operator_params"]["num_measurements"] = 200
        ebm = make_energy_model(config)
        x = torch.randn(1, 1, 32, 32)
        x_tilde = ebm.operator(x)

        assert x_tilde.shape == (1, 200)

if __name__ == "__main__":
    unittest.main()