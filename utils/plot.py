import matplotlib.pyplot as plt
import torchvision.utils as vutils
from einops import rearrange

def plot(model_samples, title, figsize=(10,30), num_of_images_per_row=5):
    plt.figure(figsize=figsize)
    img1 = vutils.make_grid(model_samples, nrow=num_of_images_per_row).cpu().detach().numpy()
    plt.title(title)
    plt.imshow(rearrange(img1, "c h w -> h w c"))
    plt.show()

def plot_samples(vaes, num=25, figsize=(10,30), num_of_images_per_row=5):
    if hasattr(vaes, '__iter__'):
        for vae in vaes:
            plot_samples(vae, num, figsize, num_of_images_per_row)
    else:
        model_samples = vaes.get_samples(num)
        title = f"Samples from {vaes.model_name}"
        plot(model_samples, title, figsize, num_of_images_per_row)
  
def plot_reconstruction(vaes, batch, num_of_samples=5, num_of_images_per_row=5, figsize=(10, 30)):
    x, y = batch
    img = x[:num_of_samples, :, :, :]
    plot(img, "Original", figsize=figsize, num_of_images_per_row=num_of_images_per_row)

    if hasattr(vaes, '__iter__'):
        for vae in vaes:
            title = f"Reconstruction from {vae.model_name}"
            plot(vae(img), title, figsize, num_of_images_per_row)
    else:
        title = f"Reconstruction from {vaes.model_name}"
        plot(vaes(img), title, figsize, num_of_images_per_row)
