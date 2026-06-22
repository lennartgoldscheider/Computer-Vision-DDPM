import time
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
from Dataloader_fertig import get_dataloader
from utils import denormalize, load_autoencoder

# This file is used to generate images via autoencoder. 
# Mainly intented to be used as export for the generate function

@torch.no_grad()
def generate_autoencoder(autoencoder_checkpoint, 
             num_images= None,
             latent_channels=4, 
             output_root="outputs/samples"):

    # find device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load models and dataloader
    autoencoder = load_autoencoder(
        autoencoder_checkpoint,
        device=device,
        latent_channels=latent_channels,
    )
    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=32,
        num_workers=0
    )

    # set and/or create sample folder
    checkpoint_name = Path(autoencoder_checkpoint).stem
    run_name = "_".join(checkpoint_name.split("_")[:3])
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("\nSaving samples to:")
    print(run_dir)

    num_images = len(dataloader.dataset) if not num_images else num_images 
    generated_images = []
    num_generations = 0
    total_start = time.perf_counter()

    for original_images in dataloader:

        # forward pass
        images = autoencoder(original_images)

        # append generated images
        generated_images.append(images.cpu())
        num_generations += len(images)

        # end loop early if enough images are generated
        if num_generations >= num_images:
            break

    total_time = (time.perf_counter() - total_start)

    # denormalize generated images and restrain number of generated images
    generated_images = torch.cat(generated_images, dim=0)
    generated_images = generated_images[:num_images]
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

    # load checkpoints
    autoencoder_checkpoint = (
        "outputs/checkpoints/"
        "flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt"
    )

    # generate images
    generate_autoencoder(
        autoencoder_checkpoint=autoencoder_checkpoint,
        num_images= 5,
        latent_channels=16,
    )


if __name__ == "__main__":
    main()