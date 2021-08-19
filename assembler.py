import os 
import importlib
from functools import partial
from utils.config import get_config_base_model, get_config_ebm
from experiment import VAEModule, EBMModule, BaseVAE

def get_base_model(config, dataset, mode):
    model_type = config["exp_params"]["base_model"]
    if model_type == "vae":
        model = make_vaes(config, dataset, mode)
    return model

def make_model(model_name):
    dataset, model, sampling, task = model_name.split("/")
    name = f"{sampling}/{task}"
    config = get_config(get_config_ebm, dataset, model, name)
    ebm = make_energy_model(config)
    return ebm

def is_mode_training(mode):
    return mode=="training"

def is_mode_inference(mode):
    return mode=="inference"

def get_config(get_config_fn, dataset, model, name):
    if not os.path.isdir('configs'):
         os.mkdir("./configs")
    fname = f"./configs/{dataset}/{model}/{name}.yaml"
    config = get_config_fn(fname)
    config["exp_params"]["file_path"] = fname
    return config

def make_and_load_base_model(model_name, use_gpu=False):
    dataset, model_type, name = model_name.split("/")
    config = get_config(get_config_base_model, dataset, model_type, name)

    model = get_base_model(config, dataset, "inference")
    if use_gpu: model = model.to("cuda")

    model.load_model()
    return model

def make_and_load_base_models(model_names: list, use_gpu=False):
  vaes = []
  for model_name in model_names:
    vae = make_and_load_base_model(model_name, use_gpu)
    vaes.append(vae)
  return vaes 

def make_gans(config, dataset_name, mode):
    pass 

def make_vaes(config, dataset_name, mode):
    # Get model components 
    base_model = config["exp_params"]["model_name"]
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

    # training vs inference time model
    if is_mode_training(mode):
        return vae 
    elif is_mode_inference(mode):
        return vae

def make_operator(config):
    operator_type = config["exp_params"]["operator"]
    operator_library = importlib.import_module(f"src.operators")
    operator_constructor = getattr(operator_library, operator_type)
    operator = operator_constructor(config)
    return operator

def make_estimator(config):
    estimator_type = config["exp_params"]["estimator"]
    estimator_library = importlib.import_module("src.sampling")
    estimator= getattr(estimator_library, f"{estimator_type}")
    return estimator

def make_energy_model(config): 
    model_name = config["base_model_params"]["model_name"]
    vae = make_and_load_base_model(model_name)
    A = make_operator(config)
    sampling_algo = make_estimator(config)
    
    vae = BaseVAE(vae)
    ebm = EBMModule(config, vae, A, sampling_algo)
    return ebm 

if __name__ == "__main__":
    # Quick test VAEs we make
    model_name = "mnist/vae/vanilla"
    model = make_and_load_base_model(model_name, use_gpu=False)
    
    # Quick make ebms
    model_name = "mnist/vae/langevin/inpainting"
    ebm = make_model(model_name)