import time
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
from Autoencoder import Autoencoder
from Dataloader import get_dataloader

def timestamp():
    return time.strftime("%d_%H%M%S")

def denormalize(x):
    return ((x.clamp(-1, 1) + 1) / 2)

# Load Autoencoder
def load_autoencoder(checkpoint_path, device, latent_channels=4, base_channels=64):

    autoencoder = Autoencoder(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    return autoencoder

# Generation
@torch.no_grad()
def generate(autoencoder_checkpoint, num_generations= None,
    latent_channels=4, output_root="outputs/samples"):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    checkpoint_name = Path(autoencoder_checkpoint).stem
    run_name = "_".join(checkpoint_name.split("_")[:3])
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to:")
    print(run_dir)


    num_generations = num_generations if num_generations else len(dataloader)
    generated = []
    total_start = time.perf_counter()

    for iter, images in enumerate(dataloader):
        print("remaining:", num_generations-iter)
        image_start = time.perf_counter()

        generated_images = autoencoder.forward(images)

        imagetime = (time.perf_counter() - image_start)

        generated.append(generated_images.cpu())

        if num_generations== iter:
            break

    total_time = (time.perf_counter() - total_start)
    generated = torch.cat(generated, dim=0)

    generated = denormalize(generated)

    # Save individual images
    for idx, image in enumerate(generated):
        save_image(image, run_dir / f"sample_{idx:04d}.png")

    # Grid of all generated images for overwiev
    grid = make_grid(generated, nrow=int(num_generations ** 0.5), normalize=False)
    save_image(grid, run_dir / "grid.png")

    # Measure average generation time
    avg_image_time = (
        total_time / num_generations
    )

    print("\nGeneration complete")
    print(f"Images: {num_generations}")
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Average/image: "
        f"{avg_image_time:.3f}s"
    )

    return generated

def main():
    autoencoder_checkpoint = (
        "outputs/checkpoints/"
        "flowers_autoencoder_epoch50_20260616_220930.pt"
    )

    generate(
        autoencoder_checkpoint=autoencoder_checkpoint,
        latent_channels=4,
        num_generations= 5,
    )


if __name__ == "__main__":
    main()