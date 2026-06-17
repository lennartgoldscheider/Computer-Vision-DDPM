import os
import time
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from Denoising import UNet
from Diffusion import GaussianDiffusion

# This file implements the generation pipeline for the DDPM model. Mainly intended to use the generate function as import.

generation_model = "outputs/checkpoints/flowers_ddpm_epoch30_20260613_1452.pt"

def timestamp():
    return time.strftime("%d_%H%M_%S")

def denormalize(x):
    # Convert [-1,1] -> [0,1]
    return ((x.clamp(-1, 1) + 1) / 2)

def load_model(checkpoint_path, device,
    image_channels=3,
    base_channels=64,
    time_emb_dim=256,):

    model = UNet(
        image_channels=image_channels,
        base_channels=base_channels,
        time_emb_dim=time_emb_dim,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def generate(
    checkpoint_path,
    num_images=16,
    batch_size=16,
    image_size=64,
    timesteps=1000,
    num_generations = None,
    schedule="cosine",
    output_root="outputs/samples",):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Device: {device}")

    model = load_model(
        checkpoint_path,
        device=device
    )

    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        schedule=schedule,
        device=device
    )

    
    checkpoint_name = Path(checkpoint_path).stem
    run_name = "_".join(checkpoint_name.split("_")[:3])
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to:")
    print(run_dir)

    generated = []
    total_start = time.perf_counter()
    remaining = num_generations if num_generations else num_images

    while remaining > 0:
        print("remaining:", remaining)
        current_batch = min(batch_size, remaining)
        batch_start = time.perf_counter()

        images = diffusion.sample(
            model=model,
            image_size=image_size,
            batch_size=current_batch,
            channels=3
        )

        batch_time = (time.perf_counter() - batch_start)

        generated.append(images.cpu())
        remaining -= current_batch

    total_time = (time.perf_counter() - total_start)
    generated = torch.cat(generated, dim=0)

    generated = denormalize(generated)

    # Save individual images
    for idx, image in enumerate(generated):
        save_image(image, run_dir / f"sample_{idx:04d}.png")

    # Grid of all generated images for overwiev
    grid = make_grid(generated, nrow=int(num_images ** 0.5), normalize=False)
    save_image(grid, run_dir / "grid.png")

    # Measure average generation time
    avg_image_time = (
        total_time / num_images
    )

    print("\nGeneration complete")
    print(f"Images: {num_images}")
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Average/image: "
        f"{avg_image_time:.3f}s"
    )

    return generated


def main():

    checkpoint_path = (generation_model)
    generate(
        checkpoint_path=checkpoint_path,
        num_images=16,
        batch_size=8,
        image_size=64,
        timesteps=1000,
    )


if __name__ == "__main__":
    main()