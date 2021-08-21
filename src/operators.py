import torch 
import numpy as np
from torch import nn
from torch.nn import functional as F

class CenterOcclude(nn.Module):

    def __init__(self, config):
        super(CenterOcclude, self).__init__()

        size = config["operator_params"]["size"] 
        batch_size = config["exp_params"]["batch_size"]
        image_shape = config["exp_params"]["image_shape"]
        self.A = self.occlude(size, [batch_size] + image_shape)
    
    def forward(self, x):
        return torch.mul(self.A, x)
    
    @staticmethod
    def occlude(size, image_shape):
        b, c, x, y = image_shape
        pad_size_x = int(np.floor((x-size)/2))
        pad_size_y = int(np.floor((y-size)/2))
        A = torch.zeros((b, c, size, size))
        if size+pad_size_x+pad_size_x < x:
            A = F.pad(A, (pad_size_x, pad_size_x+1, pad_size_y, pad_size_y+1), "constant", 1)
        else:
            A = F.pad(A, (pad_size_x, pad_size_x, pad_size_y, pad_size_y), "constant", 1)
        return A

def get_coord(sample, imgshape):
  y = sample %imgshape[2]
  x = (sample-y)//imgshape[2]
  return x,y 

class RandomOcclude(nn.Module):

    def __init__(self, config):
        super(RandomOcclude, self).__init__()

        num = config["operator_params"]["num_measurements"] 
        batch_size = config["exp_params"]["batch_size"]
        image_shape = config["exp_params"]["image_shape"]
        self.A = self.occlude(num, [batch_size] + image_shape)
    
    def forward(self, x):
        return torch.mul(self.A, x)
    
    @staticmethod
    def occlude(num, image_shape):
        b, c, x, y = image_shape
        n = x * y
        ri = np.random.choice(n, num, replace=False) # random sample of indices
        A = torch.zeros(image_shape)
        for sample in ri:
            x, y = get_coord(sample, image_shape)
            A[:, :, x, y] = 1
        return A