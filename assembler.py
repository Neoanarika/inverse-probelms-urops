import os 
import unittest
import importlib
from functools import partial
from utils.config import get_config_base_model, get_config_ebm
from experiment import GANModule, VAEModule, EBMModule, BaseVAE, BaseGAN

# Helper functions
def is_mode_training(mode):
    return mode=="training"

def is_mode_inference(mode):
    return mode=="inference"

# Unittest
class TestMake(unittest.TestCase):

    def test_make_vae(self):
        # Quick test VAEs we make
        model_name = "mnist/vae/vanilla"
        make_and_load_base_model(model_name, use_gpu=False)

    def test_make_gan(self):
        # Quick test GANs we make
        model_name = "mnist/gan/dcgan"
        make_and_load_base_model(model_name, use_gpu=False)
    
    def test_make_ebm(self):
        # Quick test VAEs we make this includes the sampler and operator code
        model_name = "mnist/vae/langevin/inpainting"
        make_ebm_model(model_name)

def get_base_model(base_model_config, dataset, mode):
    model_type = base_model_config["exp_params"]["base_model"]
    if model_type == "vae":
        model = make_vaes(base_model_config, dataset, mode)
        if is_mode_inference(mode):
            model = BaseVAE(model)
    elif model_type == "gan":
        model = make_gans(base_model_config, dataset, mode)
        if is_mode_inference(mode):
            model = BaseGAN(model)
    return model

def get_config(get_config_fn, dataset, model, name, path="."):
    if not os.path.isdir(f'{path}/configs'):
         os.mkdir(f"{path}/configs")
    fname = f"{path}/configs/{dataset}/{model}/{name}.yaml"
    config = get_config_fn(fname)
    config["exp_params"]["file_path"] = fname
    return config

def make_and_load_base_model(model_name, use_gpu=False, path="."):
    dataset, model_type, name = model_name.split("/")
    config = get_config(get_config_base_model, dataset, model_type, name, path)

    model = get_base_model(config, dataset, "inference")
    if use_gpu: model = model.to("cuda")

    model.load_model(path=path)
    return model

def make_and_load_base_models(model_names: list, use_gpu=False, path="."):
  vaes = []
  for model_name in model_names:
    vae = make_and_load_base_model(model_name, use_gpu, path)
    vaes.append(vae)
  return vaes 

def make_operator(config):
    operator_type = config["operator_params"]["operator"]
    operator_library = importlib.import_module(f"src.operators")
    operator_constructor = getattr(operator_library, operator_type)
    operator = operator_constructor(config)
    return operator

def make_estimator(config):
    estimator_type = config["estimator_params"]["estimator"]
    estimator_library = importlib.import_module("src.sampling")
    estimator= getattr(estimator_library, f"{estimator_type}")
    return estimator

def make_ebm_model(model_name):
    dataset, model, sampling, task = model_name.split("/")
    name = f"{sampling}/{task}"
    config = get_config(get_config_ebm, dataset, model, name)
    ebm = make_energy_model(config)
    return ebm

def make_gans(config, dataset_name, mode):
    # Get model components 
    base_model = config["exp_params"]["model_name"]
    components = importlib.import_module(f"models.{dataset_name}.gan")
    
    # Get component constructors
    discriminator_used = config["discriminator_params"]["discriminator_type"]
    generator_used = config["generator_params"]["generator_type"]

    discriminator = getattr(components, discriminator_used)
    generator = getattr(components, generator_used)

    GAN = getattr(importlib.import_module(f"models.{base_model}"), "GAN")

    gen = generator(config)
    disc = discriminator(config)

    # Assemble my model
    gan = GAN(base_model, gen, disc)
    gan = GANModule(config, gan)

    return gan

def make_vaes(config, dataset_name, mode):
    # Get model components 
    base_model = config["exp_params"]["model_name"]
    if "mnist" in dataset_name:
        dataset_name = "mnist"
    components = importlib.import_module(f"models.{dataset_name}.vae")

    # Get component constructors
    encoder_used = config["encoder_params"]["encoder_type"]
    decoder_used = config["decoder_params"]["decoder_type"]
    loss_used = config["loss_params"]["loss_type"]
    
    encoder = getattr(components, encoder_used)
    decoder = getattr(components, decoder_used)
    loss = getattr(components, loss_used)

    VAE = getattr(importlib.import_module(f"models.{base_model}"), "VAE")

    encoder = encoder(config)
    decoder = decoder(config)
    loss = partial(loss, config)
    
    # Assemble my model
    vae = VAE(base_model, loss, encoder, decoder)
    vae = VAEModule(config, vae)

    return vae

def make_energy_model(config, path="."): 
    model_name = config["base_model_params"]["model_name"]
    model = make_and_load_base_model(model_name, path=path)
    A = make_operator(config)
    sampling_algo = make_estimator(config)
    discriminator = None

    if config['estimator_params']['potential'] == "discriminator_weighted":
        discriminator_name = config['estimator_params']['discriminator_base_model']
        gan = make_and_load_base_model(discriminator_name, path=path)
        discriminator = gan.model.model.discriminator

    ebm = EBMModule(config, model, A, sampling_algo, discriminator=discriminator)
    return ebm 

if __name__ == "__main__":
    unittest.main()