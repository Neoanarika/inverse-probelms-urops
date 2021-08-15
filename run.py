import torch 
import argparse
from assembler import get_config, assembler
from pytorch_lightning import Trainer
from utils.dataloader import dataloader 

# Load configs
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
config = get_config(args.filename)
vae = assembler(config, "training")

# Load data
dm = dataloader(config)

# Run Training Loop
trainer= Trainer(gpus = config["trainer_params"]["gpus"],
        max_epochs = config["trainer_params"]["max_epochs"])
trainer.fit(vae, datamodule=dm)
torch.save(vae.state_dict(), f"{vae.model.name}_celeba_conv.ckpt")