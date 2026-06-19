import time
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
from Denoising import UNet
from Diffusion import GaussianDiffusion
from Autoencoder import Autoencoder, Autoencoder2
from utils import denormalize, load_diffusion_model, load_autoencoder
from Dataloader import get_dataloader

# This file is used to generate images via latent DDPM. 
# Mainly intented to be used as export for the generate function

# Generation
@torch.no_grad()
def generate(diffusion_checkpoint, autoencoder_checkpoint, 
             num_images=16,
             batch_size=8,
             latent_size=8,
             latent_channels=4,
             timesteps=1000,
             schedule="cosine",
             output_root="outputs/samples"):

    # find device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load models
    diffusion_model = load_diffusion_model(
        diffusion_checkpoint,
        device=device,
        latent_channels=latent_channels,
    )
    autoencoder = load_autoencoder(
        autoencoder_checkpoint,
        device=device,
        latent_channels=latent_channels,
    )
    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        schedule=schedule,
        device=device,
    )

    ## set and/or create sample folder
    checkpoint_name = Path(diffusion_checkpoint).stem
    run_name = "_".join(checkpoint_name.split("_")[:3])
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("\nSaving samples to:")
    print(run_dir)

    generated_images = []
    total_start = time.perf_counter()
    remaining = num_images

    # load latent stats 
    stats = torch.load("outputs/checkpoints/latent_stats.pt")
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)

    while remaining > 0:

        # get current batch
        current_batch = min(batch_size, remaining)
        batch_start = time.perf_counter()

        # Sample latents
        latents = diffusion.sample(
            model=diffusion_model,
            image_size=latent_size,
            batch_size=current_batch,
            channels=latent_channels,
        )

        ####
        dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=current_batch,
        num_workers=0
        )
        latents_all = []
        print("Preparing latent norms")
        with torch.no_grad():
            for images in dataloader:
                z = autoencoder.encode(images.to(device))
                latents_all.append(z.cpu())
                break
        ####

        # denormalze the (during training normalized) latents
        latents = latents * std + mean

        # create images
        images = autoencoder.decode(latents)

        # print progress message
        batch_time = (time.perf_counter() - batch_start)
        print(f"Generated {current_batch} images in {batch_time:.2f}s")

        # append generated images and compute number of image generations left
        generated_images.append(images.cpu())
        remaining -= current_batch

    total_time = (time.perf_counter() - total_start)

    # denormalize generated image
    generated_images = torch.cat(generated_images, dim=0)
    generated_images = denormalize(generated_images)

    # Save individual images
    for idx, image in enumerate(generated_images):
        save_image(image, run_dir / f"sample_{idx:04d}.png")

    # Grid of all generated images for overwiev
    grid = make_grid(generated_images, nrow=int(num_images ** 0.5), normalize=False)
    save_image(grid, run_dir / "grid.png")

    # Measure generation time
    avg_time = (total_time / num_images)
    print("\nGeneration complete")
    print(f"Images: {num_images}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average/image: {avg_time:.3f}s")

    return generated_images

def main():
    diffusion_checkpoint = (
        "outputs/checkpoints/"
        "latent_ddpm_epoch100_20260618_175522.pt"
    )
    autoencoder_checkpoint = (
        "outputs/checkpoints/"
        "flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt"
    )

    generate(
        diffusion_checkpoint=diffusion_checkpoint,
        autoencoder_checkpoint=autoencoder_checkpoint,
        num_images=16,
        batch_size=8,
        latent_size=8,
        latent_channels=16,
        timesteps=1000,
    )


if __name__ == "__main__":
    main()