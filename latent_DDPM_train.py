import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import time
from Dataloader_fertig import get_dataloader
from Denoising_fertig import UNet
from Diffusion_fertig import GaussianDiffusion
from Autoencoder_fertig import Autoencoder
from utils import freeze_autoencoder

# This file is used to train the latent DDPM with a given autoencoder version.

def train_latent_ddpm(model, diffusion, autoencoder, dataloader, device,
    epochs=10,
    lr=1e-4,
    log_every=20,
    save_dir = "outputs/checkpoints",
    save_every=1,
    run_name="latent_ddpm",
    timestamp = 0):

    # compute latents of the images to get 
    # mean and standard deviation for normalization
    latents_all = []
    print("Preparing latent norms")
    pbar_latent = tqdm(total= len(dataloader), leave=False)
    with torch.no_grad():
        for batch_number, images in enumerate(dataloader):
            z = autoencoder.encode(images.to(device))
            latents_all.append(z.cpu())
            pbar_latent.update(1)
            pbar_latent.set_description(f"Batch number: {batch_number+1}/{len(dataloader)}")
    latents_all = torch.cat(latents_all, dim=0)
    mean = latents_all.mean()
    std = latents_all.std()
    torch.save({"mean": mean, "std": std}, "outputs/checkpoints/latent_stats.pt")


    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    global_step = 0

    pbar = tqdm(total=epochs * len(dataloader))

    for epoch in range(epochs):
        epoch_loss = 0.0

        for x_0 in dataloader:

            # encode image
            x_0 = x_0.to(device)
            with torch.no_grad():
                latent_x_0 = autoencoder.encode(x_0)
                latent_x_0 = (latent_x_0 - mean) / (std + 1e-8)

            # get t and noise
            t = diffusion.sample_timesteps(latent_x_0.shape[0]).to(device)
            noise = torch.randn_like(latent_x_0)

            # compute x_t and noise
            latent_x_t, noise = diffusion.q_sample(latent_x_0, t, noise)

            # predict the noise epsilon_theta(x_t,t)
            pred_noise = model(latent_x_t, t)

            # compute loss and update parameters
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1

            # log losses
            if global_step % log_every == 0:
                losses.append(loss.item())

            # update progress bar
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # compute average loss per epoch
        avg_loss = epoch_loss / len(dataloader)

        # save model
        if (epoch + 1) % save_every == 0:
            model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)
        
    # final save
    model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)

    return losses


def main():

    # set autoencoder_checkpoint
    autoencoder_checkpoint = "outputs/checkpoints/flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt"

    # set parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    save_dir = "outputs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    run_name = "latent_ddpm"
    timestamp = time.strftime("%Y%m%d_%H%M")

    # set models and autoencoder
    autoencoder = Autoencoder(
        in_channels=3,
        latent_channels=16,
        base_channels=64
    ).to(device)
    model = UNet(
        image_channels=16,   # latent_channels == image_channels
        base_channels=64
    ).to(device)
    diffusion = GaussianDiffusion(
        timesteps=1000,
        schedule="cosine",
        device=device
    )
    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=4,
        num_workers=0
    )

    # load autoencoder checkpoint
    ae_ckpt = torch.load(autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(ae_ckpt["model_state_dict"])
    freeze_autoencoder(autoencoder)

    # train the model and save losses
    losses = train_latent_ddpm(
        model=model,
        diffusion=diffusion,
        autoencoder=autoencoder,
        dataloader=dataloader,
        device=device,
        epochs=200,
        lr=1e-4,
        log_every=100,
        save_every=20,
        run_name= run_name,
        timestamp=timestamp,
    )

    # create files for plotting and easy reading
    losses_save_dir = "outputs/training"
    os.makedirs(losses_save_dir, exist_ok=True)
    torch.save(losses, f"outputs/training/{run_name}_losses_{timestamp}.pt")

    print("\n✓ Losses saved to:")
    print(f"  - output/training/{run_name}_losses_{timestamp}.pt")

    loss_path_txt = os.path.join(losses_save_dir, f"{run_name}_losses_{timestamp}.txt")
    with open(loss_path_txt, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    print(f"  - {loss_path_txt}")


if __name__ == "__main__":
    main()