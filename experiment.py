import torch
import warnings
from pytorch_lightning import LightningModule
from torch.nn import functional as F


class VAEModule(LightningModule):
    """
    Standard lightning training code.
    """

    def __init__(
            self,
            model,
            lr: float = 1e-3,
    ):

        super(VAEModule, self).__init__()

        self.lr = lr
        self.model = model
        self.model_name = model.name

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def det_encode(self, x):
        x = x.to(self.device)
        mu, _ = self.model.encoder(x)
        return mu

    def stoch_encode(self, x):
        x = x.to(self.device)
        mu, log_var = self.model.encoder(x)
        z = self.model.sample(mu, log_var)
        return z

    def decode(self, z):
        return self.model.decoder(z)

    def get_samples(self, num):
        return self.model.get_samples(num)

    def step(self, batch, batch_idx):
        x, y = batch

        loss = self.model.compute_loss(x)

        logs = {
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def load_model(self):
        try:
            self.load_state_dict(torch.load(f"{self.model.name}_celeba_conv.ckpt"))
        except  FileNotFoundError:
            print(f"Please train the model using python run.py -c ./configs/{self.model.name}.yaml")