from utils.download_mnist import mnist_dataloader_train
from utils.download_celeba import celeba_dataloader 

def dataloader(config):
    # Load data
    if  config["exp_params"]["dataset"] == "celeba":
        dm = celeba_dataloader(config["exp_params"]["batch_size"],
                            config["exp_params"]["img_size"],
                            config["exp_params"]["crop_size"], 
                            config["exp_params"]["data_path"])
    elif config["exp_params"]["dataset"] == "mnist":
        dm = mnist_dataloader_train(config)
    return dm 