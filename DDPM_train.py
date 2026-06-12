# Add Diffusion process (done)
# Add Denoising process UNET (done)
# Connect Diffusion and Denoising for training (done)

# Optional: Improve training with EMA and diversity checks
# Add Generation pipeline
# Add Frechet Inception Distance & Inception Score & Average inference time for batch generation
# latent space diffusion
# Check speedup, visual quality and metrics for the accelerated version
# Ablation study for varying sampling steps T = 1000 vs 100 or 10

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Denoising import UNet
from Diffusion import GaussianDiffusion
from Dataloader import get_dataloader


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M")


def train(model, diffusion, dataloader, device,
    epochs=1,
    lr=1e-4,
    log_every=10,
    save_dir="outputs/checkpoints",
    run_name="ddpm",
    save_every_epochs=1):

    def save_checkpoint(epoch):

        timestamp = time.strftime("%Y%m%d_%H%M")

        path = os.path.join(save_dir, f"{run_name}_epoch{epoch+1}_{timestamp}.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "timestamp": timestamp
        }, path)
        print(f"\n✓ checkpoint saved → {path}")

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)

    losses = []

    pbar = tqdm(total=epochs * len(dataloader))

    for epoch in range(epochs):

        for step, x0 in enumerate(dataloader):

            x0 = x0.to(device)
            t = diffusion.sample_timesteps(x0.shape[0]).to(device)

            noise = torch.randn_like(x0)

            x_t, noise = diffusion.q_sample(x0, t, noise)
            pred_noise = model(x_t, t)

            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % log_every == 0:
                losses.append(loss.item())
                

        # ─────────────────────────────
        # SAVE CHECKPOINT (per epoch)
        # ─────────────────────────────
        if (epoch + 1) % save_every_epochs == 0:
            save_checkpoint(epoch)

    # final save ALWAYS
    save_checkpoint(epochs-1)

    return losses

import os
import numpy as np
import torch

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet()

    diffusion = GaussianDiffusion(
        timesteps=1000,
        schedule="cosine",
        device=device
    )

    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=8,
        num_workers=0
    )

    save_dir = "outputs/checkpoints"
    run_name = "flowers_ddpm"

    losses = train(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        device=device,
        epochs=3,
        log_every=20,
        run_name=run_name,
        save_dir=save_dir,
        save_every_epochs=20
    )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M")

    loss_path_npy = os.path.join(save_dir, f"{run_name}_losses_{timestamp}.npy")
    loss_path_txt = os.path.join(save_dir, f"{run_name}_losses_{timestamp}.txt")

    # binary (best for later plotting)
    np.save(loss_path_npy, np.array(losses))

    # human-readable backup
    with open(loss_path_txt, "w") as f:
        for l in losses:
            f.write(f"{l}\n")

    print(f"\n✓ Losses saved to:")
    print(f"  - {loss_path_npy}")
    print(f"  - {loss_path_txt}")


if __name__ == "__main__":
    main()
