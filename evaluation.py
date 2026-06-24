import torch
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from data import load_images
from globals import *


# compute Metrics
def evaluate_metrics(real_dir, fake_dir, device="cuda"):

    # load real and fake (generated) images
    real_loader = load_images(real_dir) 
    fake_loader = load_images(fake_dir) 

    # initialize Fréchet Inception Distance (FID) and Inception Score (IS) 
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # REAL IMAGES
    pbar_real = tqdm(real_loader, leave=False)
    for imgs in pbar_real:
        imgs = imgs.to(device)
        fid.update(imgs, real=True)
    print("Real Images done")

    # FAKE IMAGES
    pbar_fake = tqdm(fake_loader, leave=False)
    for imgs in pbar_fake:
        imgs = imgs.to(device)
        fid.update(imgs, real=False)
        inception.update(imgs)
    print("Fake Images done")

    # compute FID score and mean/ std of IS
    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()

    return fid_score, is_mean.item(), is_std.item()

def main():

    # path to directories
    real_dir = REAL_DIR
    fake_dirs = LIST_SAMPLES_DIR
    
    for fake_dir in fake_dirs:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Evaluating...")

        # compute metrics
        fid, is_mean, is_std = evaluate_metrics(real_dir=real_dir, fake_dir=fake_dir, device=device)

        print("\n--- Results ---")
        print(f"FID: {fid:.2f}")
        print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")


if __name__ == "__main__":
    main()

# --- Results DDPM ---
# FID: 219.87
# Inception Score: 3.02 ± 0.11


# --- Results Latent DDPM 16 channels ---
# FID: 114.06
# Inception Score: 3.85 ± 0.10

# --- Results Latent DDPM 8 channels ---
# FID: 76.88
# Inception Score: 2.85 ± 0.15

# --- Results Latent DDPM 4 channels ---
# FID: 103.45
# Inception Score: 2.61 ± 0.04


