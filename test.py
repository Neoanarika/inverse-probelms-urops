import torch 
import unittest
from models.mnist.vae import ConvEncoder, ConvDecoder
from models.mnist.gan import DCGANGenerator, DCGANDiscriminator
from utils.config import get_config_base_model
from assembler import make_energy_model, get_config, get_config_ebm

def get_model_config(model_name):
    dataset, model, sampling, task = model_name.split("/")
    name = f"{sampling}/{task}"
    config = get_config(get_config_ebm, dataset, model, name)
    return config

# Unittest
class TestVAE(unittest.TestCase):

    def test_conv_encoder(self):
        config = get_config_base_model("./configs/mnist/vae/vanilla.yaml")
        enc = ConvEncoder(config)
        img = torch.randn(1, 1, 32, 32)
        mu, log_var = enc(img)
        assert mu.shape == (1, config["exp_params"]["latent_dim"])
        assert log_var.shape == (1, config["exp_params"]["latent_dim"])

    def test_conv_decoder(self):
        config = get_config_base_model("./configs/mnist/vae/vanilla.yaml")
        dec = ConvDecoder(config)
        noise = torch.randn(1, config["exp_params"]["latent_dim"])
        img = dec(noise)
        assert img.shape == (1, 1, 32, 32)

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
        pred = disc(img)
        assert pred.shape == (1, )

class TestSampler(unittest.TestCase):

    def test_langevin(self):
        config = get_model_config("mnist/vae/langevin/inpainting")
        ebm = make_energy_model(config)
        x_tilde = torch.randn(1, 1, 32, 32)
        x_hat = ebm(x_tilde)
        assert x_hat.shape == (1,  1, 32, 32)
    
    def test_map(self):
        config = get_model_config("mnist/vae/langevin/inpainting")
        config["estimator_params"]["estimator"] = "map"
        config["estimator_params"]["n"] = 15
        ebm = make_energy_model(config)
        x_tilde = torch.randn(1, 1, 32, 32)
        x_hat = ebm(x_tilde)
        assert x_hat.shape == (1,  1, 32, 32)

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
    
    def test_guassian_noise(self):
        model_name = "mnist/vae/langevin/inpainting"
        config = get_model_config(model_name)
        config["operator_params"]["operator"] = "GuassianNoise"
        config["operator_params"]["noise_level"] = 1
        ebm = make_energy_model(config)
        x = torch.randn(1, 1, 32, 32)
        x_tilde = ebm.operator(x)

        assert x_tilde.shape == (1,  1, 32, 32)


if __name__ == "__main__":
    unittest.main()