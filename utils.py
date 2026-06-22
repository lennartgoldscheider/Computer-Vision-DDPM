import torch
from Denoising_fertig import UNet
from Autoencoder_fertig import Autoencoder
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Convert [-1,1] to [0,1]
def denormalize(x):
    return ((x.clamp(-1, 1) + 1) / 2)

# Load Latent DDPM
def load_diffusion_model(checkpoint_path, device,
                         image_channels=3,
                         base_channels=64,
                         time_emb_dim=256):

    # initialize UNet
    model = UNet(
        image_channels=image_channels,
        base_channels=base_channels,
        time_emb_dim=time_emb_dim,
    ).to(device)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# Load Autoencoder
def load_autoencoder(checkpoint_path, device, 
                     latent_channels=4,
                     base_channels=64):

    # initialize Autoencoder
    autoencoder = Autoencoder(
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

# Dataset (real images) for evaluate.py
class ImageFolderDataset(Dataset):
    def __init__(self, root, 
                 image_size=64):
        self.root = Path(root)
        self.paths = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.PILToTensor(),  
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# for evaluate.py
def load_images(folder, image_size=64, batch_size=32):
    dataset = ImageFolderDataset(folder, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
