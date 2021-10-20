import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from utils.config import get_config_base_model
from torch.distributions.bernoulli import Bernoulli

def download_mnist(config):
    MNIST(f'{config["exp_params"]["data_path"]}', train=True, download=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    MNIST(f'{config["exp_params"]["data_path"]}', train=False, download=True,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))

def mnist_dataloader_train(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_train = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=True, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    dm = DataLoader(mnist_train, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def mnist_dataloader_test(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_test = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=False, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32))]))
    dm = DataLoader(mnist_test, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def add_gradient(x):
    y = torch.linspace(0, 0.5, steps=32)
    filter_mat = y.repeat(32,1).t()
    return 0.5*x + filter_mat

def add_multi_gradient(x):
    m = Bernoulli(torch.tensor([0.5]))
    u = m.sample()
    y = torch.linspace(0, 0.5, steps=32) if u ==0  else torch.linspace(0.5, 0, steps=32)
    filter_mat = y.repeat(32,1).t()
    return 0.5*x + filter_mat

def add_multi_gradient2(x):
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    m = torch.distributions.Categorical(probs)
    u = m.sample()
    if u==0:
        y = torch.linspace(0, 0.5, steps=32) 
        filter_mat = y.repeat(32,1).t()
    elif u == 1:
        y = torch.linspace(0.5, 0, steps=32)
        filter_mat = y.repeat(32,1).t()
    elif u == 2:
        y = torch.linspace(0, 0.5, steps=32)
        filter_mat = y.repeat(32,1)
    elif u == 3:
        y = torch.linspace(0.5, 0, steps=32)
        filter_mat = y.repeat(32,1)
    return 0.5*x + filter_mat

def modified_mnist_dataloader_train(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_train = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=True, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32)),
                                                       lambda x: add_gradient(x)
                                                       ]))
    dm = DataLoader(mnist_train, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def multi_mnist_dataloader_train(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_train = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=True, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32)),
                                                       lambda x: add_multi_gradient(x)
                                                       ]))
    dm = DataLoader(mnist_train, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def multi_two_mnist_dataloader_train(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_train = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=True, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32)),
                                                       lambda x: add_multi_gradient2(x)
                                                       ]))
    dm = DataLoader(mnist_train, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def modified_mnist_dataloader_test(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_test = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=False, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32)),
                                                       lambda x: add_gradient(x)]))
    dm = DataLoader(mnist_test, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def multi_mnist_dataloader_test(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_test = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=False, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32)),
                                                       lambda x: add_multi_gradient(x)]))
    dm = DataLoader(mnist_test, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

def multi_two_mnist_dataloader_test(config, path="", shuffle=False):
    path = "" if path == "" else path+"/"
    mnist_test = MNIST(f'{path}{config["exp_params"]["data_path"]}', train=False, download=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Resize((32,32)),
                                                       lambda x: add_multi_gradient2(x)]))
    dm = DataLoader(mnist_test, batch_size=config["exp_params"]["batch_size"], shuffle=shuffle)
    return dm

if __name__ == "__main__":
    fpath= f"./configs/vanilla_vae_mnist.yaml"
    config = get_config_base_model(fpath)
    download_mnist(config)