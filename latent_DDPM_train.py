import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import time

from Dataloader import get_dataloader
from Denoising import UNet
from Diffusion import GaussianDiffusion
from Autoencoder import Autoencoder


# --------------------------------------------------
# freeze autoencoder
# --------------------------------------------------

def freeze_autoencoder(autoencoder):
    autoencoder.eval()

    for p in autoencoder.parameters():
        p.requires_grad = False


# --------------------------------------------------
# training loop
# --------------------------------------------------

def train_latent_ddpm(
    model,
    diffusion,
    autoencoder,
    dataloader,
    device,
    epochs=10,
    lr=1e-4,
    log_every=20,
    save_every=1,
    run_name="latent_ddpm"
):

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    losses = []

    total_steps = epochs * len(dataloader)

    global_step = 0

    pbar = tqdm(total=total_steps, desc="Training", leave=True)

    for epoch in range(epochs):

        epoch_loss = 0.0

        for step, images in enumerate(dataloader):

            images = images.to(device)

            # --------------------------------------------------
            # 1. encode image → latent
            # --------------------------------------------------
            with torch.no_grad():
                latents = autoencoder.encode(images)

            # --------------------------------------------------
            # 2. sample timestep
            # --------------------------------------------------
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            noise = torch.randn_like(latents)

            # --------------------------------------------------
            # 3. forward diffusion in latent space
            # --------------------------------------------------
            x_t, noise = diffusion.q_sample(latents, t, noise)

            # --------------------------------------------------
            # 4. predict noise
            # --------------------------------------------------
            pred_noise = model(x_t, t)

            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --------------------------------------------------
            # logging
            # --------------------------------------------------
            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                losses.append(loss.item())

            pbar.set_postfix(
                epoch=f"{epoch+1}/{epochs}",
                loss=f"{loss.item():.4f}"
            )
            pbar.update(1)

        avg_loss = epoch_loss / len(dataloader)

        print(f"\nEpoch {epoch+1} | loss {avg_loss:.6f}")

        # --------------------------------------------------
        # save checkpoint
        # --------------------------------------------------
        if (epoch + 1) % save_every == 0:

            ckpt_path = (
                checkpoint_dir /
                f"{run_name}_epoch{epoch+1}_{timestamp}.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss
                },
                ckpt_path
            )

            print(f"Saved: {ckpt_path}")

    pbar.close()

    return losses


# --------------------------------------------------
# main
# --------------------------------------------------

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # --------------------------------------------------
    # autoencoder (frozen)
    # --------------------------------------------------
    autoencoder = Autoencoder(
        in_channels=3,
        latent_channels=4,
        base_channels=64
    ).to(device)

    ae_ckpt = torch.load(
        "outputs/checkpoints/flowers_autoencoder_epoch50_20260614_171314.pt",
        map_location=device
    )

    autoencoder.load_state_dict(ae_ckpt["model_state_dict"])

    freeze_autoencoder(autoencoder)

    # --------------------------------------------------
    # diffusion model (UNet must match latent channels)
    # --------------------------------------------------
    model = UNet(
        image_channels=4,   # IMPORTANT: latent channels
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
        batch_size=8,
        num_workers=0
    )

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
        run_name="latent_ddpm"
    )

    torch.save(losses, "training/latent_ddpm_losses.pt")


if __name__ == "__main__":
    main()