from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.config import get_config_base_model

def download_mnist(config):
    MNIST(f'{config["exp_params"]["data_path"]}', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    MNIST(f'{config["exp_params"]["data_path"]}', train=False, download=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))

def mnist_dataloader_train(config):
    mnist_train = MNIST(f'{config["exp_params"]["data_path"]}', train=True, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    dm = DataLoader(mnist_train, batch_size=config["exp_params"]["batch_size"])
    return dm

def mnist_dataloader_test(config, path=""):
    mnist_test = MNIST(f'{path}/{config["exp_params"]["data_path"]}', train=False, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    dm = DataLoader(mnist_test, batch_size=config["exp_params"]["batch_size"])
    return dm

if __name__ == "__main__":
    fpath= f"./configs/vanilla_vae_mnist.yaml"
    config = get_config_base_model(fpath)
    download_mnist(config)