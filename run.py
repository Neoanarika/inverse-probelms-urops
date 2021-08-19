import torch 
import argparse
from assembler import get_base_model
from utils.config import get_config_base_model, get_config_hash
from pytorch_lightning import Trainer
from utils.dataloader import dataloader 

# Load configs
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--model',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
config = get_config_base_model(args.filename)
vae = get_base_model(config, "mnist", "training")

# Load data
dm = dataloader(config)

# Run Training Loop
trainer= Trainer(gpus = config["trainer_params"]["gpus"],
        max_epochs = config["trainer_params"]["max_epochs"])
trainer.fit(vae, dm)
hash = get_config_hash(config)
torch.save(vae.state_dict(), f'./{config["exp_params"]["checkpoint_path"]}/{hash}.ckpt')