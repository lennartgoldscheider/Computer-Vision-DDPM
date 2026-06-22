import time
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
from Diffusion import GaussianDiffusion
from utils import denormalize, load_diffusion_model

# This file implements the generation pipeline for the DDPM model.
# Mainly intended to use the generate function as import.

@torch.no_grad()
def generate_DDPM(checkpoint_path,
    num_images=16,
    batch_size=16,
    image_size=64,
    timesteps=1000,
    schedule="cosine",
    output_root="outputs/samples",):

    # find device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load models
    diffusion_model = load_diffusion_model(
        checkpoint_path,
        device=device
    )
    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        schedule=schedule,
        device=device
    )

    # set and/or create sample folder
    checkpoint_name = Path(checkpoint_path).stem
    run_name = "_".join(checkpoint_name.split("_")[:3])
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Saving samples to:")
    print(run_dir)

    generated = []
    total_start = time.perf_counter()
    remaining = num_images

    while remaining > 0:

        # get current batch
        current_batch = min(batch_size, remaining)
        batch_start = time.perf_counter()

        # create images
        images = diffusion.sample(
            model=diffusion_model,
            image_size=image_size,
            batch_size=current_batch,
            channels=3
        )

        # print progress message
        batch_time = (time.perf_counter() - batch_start)
        print(f"Generated {current_batch} images in {batch_time:.2f}s")

        # append generated images and compute number of image generations left
        generated.append(images.cpu())
        remaining -= current_batch

    total_time = (time.perf_counter() - total_start)

    # denormalize generated image
    generated = torch.cat(generated, dim=0)
    generated = denormalize(generated)

    # Save individual images
    for idx, image in enumerate(generated):
        save_image(image, run_dir / f"sample_{idx:04d}.png")

    # Grid of all generated images for overwiev
    grid = make_grid(generated, nrow=int(num_images ** 0.5), normalize=False)
    save_image(grid, run_dir / "grid.png")

    # Measure average generation time
    avg_image_time = (total_time / num_images)
    print("\nGeneration complete")
    print(f"Images: {num_images}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average/image: {avg_image_time:.3f}s")

    return generated


def main():

    checkpoint_path = "outputs/checkpoints/flowers_ddpm_epoch1_20260618_1640.pt"
    generate_DDPM(
        checkpoint_path=checkpoint_path,
        num_images=16,
        batch_size=32,
        image_size=64,
        timesteps=1000,
    )


if __name__ == "__main__":
    main()