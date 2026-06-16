import time
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


# Dataset (real images)
class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size=64):
        self.root = Path(root)
        self.paths = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.PILToTensor(),  # uint8 [0,255]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def load_images(folder, image_size=64, batch_size=32):
    dataset = ImageFolderDataset(folder, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Metrics
def evaluate_metrics(real_dir, fake_dir, device="cuda"):
    real_loader = load_images(real_dir)
    fake_loader = load_images(fake_dir)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # REAL IMAGES
    for imgs in real_loader:
        imgs = imgs.to(device)
        fid.update(imgs, real=True)

    # FAKE IMAGES
    for imgs in fake_loader:
        imgs = imgs.to(device)
        fid.update(imgs, real=False)
        inception.update(imgs)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()

    return fid_score, is_mean.item(), is_std.item()

def main():

    real_dir = "datasets/flowers"
    fake_dirs = ["outputs/samples/flowers_ddpm_epoch20", "outputs/samples/flowers_ddpm_epoch180", "outputs/samples/latent_ddpm_epoch20", "outputs/samples/latent_ddpm_epoch180"]
    for fake_dir in fake_dirs:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Evaluating...")

        fid, is_mean, is_std = evaluate_metrics(real_dir=real_dir, fake_dir=fake_dir, device=device)

        print("\n--- Results ---")
        print(f"FID: {fid:.2f}")
        print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")


if __name__ == "__main__":
    main()

# --- Results ---
# FID: 423.27
# Inception Score: 1.31 ± 0.24
# Evaluating...

# --- Results ---
# FID: 235.21
# Inception Score: 1.54 ± 0.24
# Evaluating...

# --- Results ---
# FID: 385.30
# Inception Score: 1.45 ± 0.28
# Evaluating...

# --- Results ---
# FID: 266.05
# Inception Score: 1.46 ± 0.30