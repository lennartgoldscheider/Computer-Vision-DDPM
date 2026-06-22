from pathlib import Path
from latent_DDPM_generate_fertig import generate_latent_DDPM

# Generation of images for all Latent DDPM checkpoints

CHECKPOINT_DIR = Path("outputs/checkpoints")


def find_checkpoint(epoch):
    pattern = f"latent_ddpm_epoch{epoch}_*.pt"
    matches = sorted(CHECKPOINT_DIR.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")

    if len(matches) > 1:
        print(f"Warning: multiple checkpoints found for epoch {epoch}. Using newest.")
        
    return matches[-1]


def generate_from_epochs(epochs, autoencoder_checkpoint,
    num_images=16,
    batch_size=8,
    latent_size=8,
    latent_channels=4,
    timesteps=1000):

    for epoch in epochs:
        checkpoint = find_checkpoint(epoch)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}")
        print(checkpoint.name)
        print("=" * 60)

        generate_latent_DDPM(
            diffusion_checkpoint=str(checkpoint),
            autoencoder_checkpoint=autoencoder_checkpoint,
            num_images=num_images,
            batch_size=batch_size,
            latent_size=latent_size,
            latent_channels=latent_channels,
            timesteps=timesteps,
        )


def main():
    autoencoder_checkpoint = (
        "outputs/checkpoints/"
        "flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt"
    )

    epochs = [100] # [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    generate_from_epochs(
        epochs=epochs,
        autoencoder_checkpoint=autoencoder_checkpoint,
        num_images=16,
        batch_size=8,
        latent_size=8,
        latent_channels=16,
        timesteps=1000,
    )


if __name__ == "__main__":
    main()