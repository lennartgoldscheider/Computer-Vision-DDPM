import time
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
from data import get_dataloader
from utils import denormalize, load_autoencoder, load_diffusion_model, GaussianDiffusion
from globals import *

# ------------------------------------------
# ------- Autoencoder Reconstruction -------
# ------------------------------------------

@torch.no_grad()
def generate_autoencoder(autoencoder, dataloader,
             num_images= None,
             batch_size=16,
             ):

    num_images = len(dataloader.dataset) if not num_images else num_images 
    generated_images = []
    num_generations = 0
    total_start = time.perf_counter()

    for original_images in dataloader:

        batch_start = time.perf_counter()

        # forward pass
        images = autoencoder(original_images)

        # append generated images
        generated_images.append(images.cpu())
        num_generations += len(images)

        # print progress message
        batch_time = (time.perf_counter() - batch_start)
        print(f"Generated {batch_size} images in {batch_time:.2f}s")

        # end loop early if enough images are generated
        if num_generations >= num_images:
            break

    total_time = (time.perf_counter() - total_start)

    # denormalize generated images and restrain number of generated images
    generated_images = torch.cat(generated_images, dim=0)
    generated_images = generated_images[:num_images]
    generated_images = denormalize(generated_images)

    return generated_images, total_time

# -----------------------------------------
# ------------ DDPM Generation ------------
# -----------------------------------------

@torch.no_grad()
def generate_DDPM(diffusion_model, diffusion,
    num_images=16,
    batch_size=16,
    image_size=64,
    latent= False, # only if latent DDPM
    mean = 0, # only if latent DDPM
    std = 0, # only if latent DDPM
    latent_size = 0, # only if latent DDPM
    latent_channels = 0, # only if latent DDPM
    autoencoder = None, # only if latent DDPM
    ):

    generated_images = []
    total_start = time.perf_counter()
    remaining = num_images

    while remaining > 0:

        # get current batch
        current_batch = min(batch_size, remaining)
        batch_start = time.perf_counter()

        # create images
        if latent:
            # Sample latents
            latents = diffusion.sample(
                model=diffusion_model,
                image_size=latent_size,
                batch_size=current_batch,
                channels=latent_channels,
            )

            # denormalze the (during training normalized) latents
            latents = latents * std + mean

            images = autoencoder.decode(latents)

        else:
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
        generated_images.append(images.cpu())
        remaining -= current_batch
        
    total_time = (time.perf_counter() - total_start)

    # denormalize generated image
    generated_images = torch.cat(generated_images, dim=0)
    generated_images = denormalize(generated_images)

    return generated_images, total_time

# --------------------------------------
# ---------- MAIN for Generating -------
# --------------------------------------
def main():

    # find device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"generating with {GENERATING_ALGORITHM}")

    # set and/or create sample folder
    checkpoint_path = AUTOENCODER_GENERATING_CHECKPOINT if GENERATING_ALGORITHM == "Autoencoder" else DDPM_CHECKPOINT if GENERATING_ALGORITHM == "DDPM" else LATENT_DDPM_CHECKPOINT
    checkpoint_name = Path(checkpoint_path).stem
    run_name = "_".join(checkpoint_name.split("_")[:4])
    run_dir = Path(OUTPUT_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Saving samples to:")
    print(run_dir)

    
    # ------------ Autoencoder ------------
    if GENERATING_ALGORITHM =="Autoencoder":

        # load models and dataloader
        autoencoder = load_autoencoder(
            AUTOENCODER_GENERATING_CHECKPOINT,
            device=device,
            latent_channels=LATENT_CHANNELS_GENERATING,
        )
        dataloader = get_dataloader(
            root="datasets/flowers",
            image_size=IMAGE_SIZE_GENERATING,
            batch_size=BATCH_SIZE_GENERATING,
            num_workers=0
        )

        # generate images
        generated_images, total_time = generate_autoencoder(
            autoencoder = autoencoder,
            dataloader = dataloader,
            num_images = NUM_IMAGES_GENERATING,
            batch_size = BATCH_SIZE_GENERATING,
        )


    # ---------------- DDPM ----------------     
    elif GENERATING_ALGORITHM == "DDPM":

        # load models
        diffusion_model = load_diffusion_model(
            DDPM_CHECKPOINT,
            device=device
        )
        diffusion = GaussianDiffusion(
            timesteps=TIMESTEPS_GENERATING,
            schedule=SCHEDULE_GENERATING,
            device=device
        )
        
        # generate images
        generated_images, total_time = generate_DDPM(
            diffusion_model = diffusion_model,
            diffusion = diffusion,
            device = device,
            num_images = NUM_IMAGES_GENERATING,
            batch_size = BATCH_SIZE_GENERATING,
            image_size = IMAGE_SIZE_GENERATING,
        )


    # ------------- Latent DDPM -------------
    elif GENERATING_ALGORITHM == "Latent DDPM":

        # load models
        diffusion_model = load_diffusion_model(
            LATENT_DDPM_CHECKPOINT,
            device=device,
            image_channels=LATENT_CHANNELS_GENERATING,
        )
        autoencoder = load_autoencoder(
            AUTOENCODER_GENERATING_CHECKPOINT,
            device=device,
            latent_channels=LATENT_CHANNELS_GENERATING,
        )
        diffusion = GaussianDiffusion(
            timesteps=TIMESTEPS_GENERATING,
            schedule=SCHEDULE_GENERATING,
            device=device,
        )

        # load latent stats 
        stats = torch.load(f"outputs/checkpoints/{run_name}_stats.pt")
        mean = stats["mean"].to(device)
        std = stats["std"].to(device)

        # generate images
        generated_images, total_time = generate_DDPM(
            diffusion_model = diffusion_model,
            diffusion = diffusion,
            autoencoder = autoencoder,
            num_images = NUM_IMAGES_GENERATING,
            batch_size = BATCH_SIZE_GENERATING,
            latent_size = LATENT_SIZE,
            latent_channels = LATENT_CHANNELS_GENERATING,
            latent= True,
            mean = mean,
            std = std,
            )
    else:
        print("'GENERATING_ALGORITHM' specified in 'globals.py' is not valid")



    # Save individual images
    for idx, image in enumerate(generated_images):
        save_image(image, run_dir / f"sample_{idx:04d}.png")

    # Grid of all generated images for overwiev
    grid = make_grid(generated_images, nrow=int(NUM_IMAGES_GENERATING ** 0.5), normalize=False)
    save_image(grid, run_dir / "grid.png")

    # Measure generation time
    avg_time = (total_time / NUM_IMAGES_GENERATING)
    print("\nGeneration complete")
    print(f"Images: {NUM_IMAGES_GENERATING}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average/image: {avg_time:.3f}s")


if __name__ == "__main__":
    main()