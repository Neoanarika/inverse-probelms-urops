from utils.download_mnist import mnist_dataloader_test
from assembler import get_config, get_config_ebm, make_energy_model

def get_model_config(model_name):
    dataset, model, sampling, task = model_name.split("/")
    name = f"{sampling}/{task}"
    config = get_config(get_config_ebm, dataset, model, name)
    return config

model_name = "mnist/vae/langevin/inpainting"
config = get_model_config(model_name)
ebm = make_energy_model(config)

dm = mnist_dataloader_test(config)

batch = next(iter(dm))

x, y = batch
ebm(x)