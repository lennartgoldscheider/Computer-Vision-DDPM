import torch
from utils import load_images
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


# compute Metrics
def evaluate_metrics(real_dir, fake_dir, device="cuda"):

    # load real and fake (generated) images
    real_loader = load_images(real_dir) #get_dataloader(root=real_dir, batch_size=32, num_workers=0,
                #shuffle= False, pin_memory=False, drop_last = False) #load_images(real_dir)
    fake_loader = load_images(fake_dir) #get_dataloader(root=fake_dir, batch_size=32, num_workers=0,
                #shuffle= False, pin_memory=False, drop_last = False) 

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
    real_dir = "datasets/flowers"
    fake_dirs = ["outputs/samples/latent_ddpm_epoch100"]
    #["outputs/samples/flowers_ddpm_epoch20", "outputs/samples/flowers_ddpm_epoch180", "outputs/samples/latent_ddpm_epoch20", "outputs/samples/latent_ddpm_epoch180"]
    
    
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