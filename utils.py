import time
import torch
from Denoising import UNet
from Diffusion import GaussianDiffusion
from Autoencoder import Autoencoder, Autoencoder2

## unnecessary

def get_timestamp():
    return time.strftime("%d_%H%M")


## necessary

def denormalize(x):
    # Convert [-1,1] -> [0,1]
    return ((x.clamp(-1, 1) + 1) / 2)

# Load Latent DDPM
def load_diffusion_model(checkpoint_path, device,
                        latent_channels=4,
                        base_channels=64,
                        time_emb_dim=256):

    # initialize UNet
    model = UNet(
        image_channels=latent_channels,
        base_channels=base_channels,
        time_emb_dim=time_emb_dim,
    ).to(device)

    # Load model
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# Load Autoencoder
def load_autoencoder(checkpoint_path, device, 
                     latent_channels=4,
                     base_channels=64):

    # initialize Autoencoder
    autoencoder = Autoencoder2(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    # load Autoencoder
    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    return autoencoder

def freeze_autoencoder(autoencoder):
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
