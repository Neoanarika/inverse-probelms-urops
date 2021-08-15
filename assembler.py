import importlib
from functools import partial
from models.guassian_vae import VAE
from experiment import VAEModule
from utils.config import get_config

def is_config_valid(config):
    # Check config is valid
    assert type(config["exp_params"]["model_name"]) == str 
    assert type(config["exp_params"]["template"]) == str
    assert config["encoder_params"]["latent_dim"] == config["decoder_params"]["latent_dim"]

def is_mode_training(mode):
    return mode=="training"

def is_mode_inference(mode):
    return mode=="inference"

def assembler(config, mode):
    # Get model name
    is_config_valid(config)

    # Get model components 
    vae_name = config["exp_params"]["model_name"]
    dataset_name = config["exp_params"]["dataset"]
    componets = importlib.import_module(f"models.{dataset_name}.{vae_name}")
    encoder = componets.Encoder(**config["encoder_params"])
    decoder = componets.Decoder(**config["decoder_params"])
    loss = partial(componets.loss, config["loss_params"])
    
    # Assemble my model
    vae = VAE(vae_name, loss, encoder, decoder)
    vae = VAEModule(vae, config["exp_params"]["LR"])

    # training vs inference time model
    if is_mode_training(mode):
        return vae 
    elif is_mode_inference(mode):
        return vae

if __name__ == "__main__":
    fpath= f"./configs/vanilla_vae_mnist.yaml"
    config = get_config(fpath)
    vae = assembler(config, "training")
    vae = assembler(config, "inference")