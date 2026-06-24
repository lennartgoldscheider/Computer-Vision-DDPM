import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network import Autoencoder, UNet
from data import get_dataloader
from utils import GaussianDiffusion, load_autoencoder, freeze_autoencoder
from globals import *


# ------------------------------------------
# ---------- Autoencoder Training ----------
# ------------------------------------------

def train_autoencoder(model, dataloader, device,
    epochs=50,
    lr=1e-4,
    log_every=1,
    save_dir = "outputs/checkpoints",
    save_every=10,
    run_name="autoencoder",
    timestamp = 0):

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_loss = 0.0

        for step, images in enumerate(pbar):

            images = images.to(device)

            # reconstruct images
            reconstructed = model(images)

            # compute loss and update parameters
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % log_every == 0:
                losses.append(loss.item())

            # update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            break
        # compute average loss per epoch
        avg_loss = epoch_loss / len(dataloader)

        # print in between loss
        print(f"Epoch {epoch+1:03d} | avg loss = {avg_loss:.6f}")

        # save model
        if (epoch + 1) % save_every == 0:
            model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)
        break
    # final save
    model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)

    return losses

# -----------------------------------
# ---------- DDPM Training ----------
# -----------------------------------

def train_ddpm(model, diffusion, dataloader, device,
    epochs=1,
    lr=1e-4,
    log_every=10,
    save_dir="outputs/checkpoints",
    run_name="ddpm",
    save_every=1,
    timestamp = 0,
    latent= False, # only if latent DDPM
    mean = 0, # only if latent DDPM
    std=0, # only if latent DDPM
    autoencoder = None, # only if latent DDPM
    ):

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    pbar = tqdm(total=epochs * len(dataloader)) # progress bar

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, x_0 in enumerate(dataloader):

            x_0 = x_0.to(device)
            # transform into latent space if wanted
            if latent:
                with torch.no_grad():
                    x_0 = autoencoder.encode(x_0)
                    x_0 = (x_0 - mean) / (std + 1e-8)

            # get t and noise
            t = diffusion.sample_timesteps(x_0.shape[0]).to(device)
            noise = torch.randn_like(x_0)

            # compute x_t and noise
            x_t, noise = diffusion.q_sample(x_0, t, noise)

            # predict the noise epsilon_theta(x_t,t)
            pred_noise = model(x_t, t)

            # compute loss and update parameters
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # update progress bar
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # log the loss
            if step % log_every == 0:
                losses.append(loss.item())
            break
        # compute average loss per epoch
        avg_loss = epoch_loss / len(dataloader)

        # save model
        if (epoch + 1) % save_every == 0:
            model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)
        break
    # final save
    model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)

    return losses



# -----------------------------------
# ---------- MAIN for Training ------
# -----------------------------------

def main():

    # set parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M")

    print(f"Training {TRAINING_ALGORITHM}")

    # set dataloader
    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=IMAGE_SIZE_TRAINING,
        batch_size=BATCH_SIZE_TRAINING,
        num_workers=0
    )

    if TRAINING_ALGORITHM =="Autoencoder":
        model = Autoencoder(
            in_channels=3,
            latent_channels=LATENT_CHANNELS_TRAINING,
            base_channels=BASE_CHANNELS_TRAINING,
        ).to(device)

        # train the model and save losses
        losses = train_autoencoder(
            model=model,
            dataloader=dataloader,
            device=device,
            epochs=EPOCHS,
            lr = LEARNING_RATE,
            log_every=LOG_EVERY,
            save_every=SAVE_EVERY,
            run_name=RUN_NAME,
            save_dir = SAVE_DIR,
            timestamp= timestamp,
        )

    else:
        diffusion = GaussianDiffusion(
                timesteps=TIMESTEPS_TRAINING,
                schedule=SCHEDULE_TRAINING,
                device=device
            )
    
        if TRAINING_ALGORITHM == "DDPM":
            model = UNet()

            # train the model and save losses
            losses = train_ddpm(
                model=model,
                diffusion=diffusion,
                dataloader=dataloader,
                device=device,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                log_every=LOG_EVERY,
                save_every=SAVE_EVERY,
                run_name=RUN_NAME,
                save_dir=SAVE_DIR,
                timestamp = timestamp,
            )

        elif TRAINING_ALGORITHM =="Latent DDPM":
            # set autoencoder_checkpoint
            autoencoder_checkpoint = AUTOENCODER_TRAINING_CHECKPOINT
            
            autoencoder = load_autoencoder(autoencoder_checkpoint, 
                                        device, 
                                        latent_channels=LATENT_CHANNELS_TRAINING, 
                                        base_channels=BASE_CHANNELS_TRAINING).to(device)
            model = UNet(
                image_channels=LATENT_CHANNELS_TRAINING,
                base_channels=BASE_CHANNELS_TRAINING,
            ).to(device)

            # freeze autoencoder
            freeze_autoencoder(autoencoder)

            # compute latents of the images to get 
            # mean and standard deviation for normalization
            latents_all = []
            print("Preparing latent norms")
            pbar_latent = tqdm(total= len(dataloader), leave=True)
            with torch.no_grad():
                for batch_number, images in enumerate(dataloader):
                    z = autoencoder.encode(images.to(device))
                    latents_all.append(z.cpu())
                    pbar_latent.update(1)
                    pbar_latent.set_description(f"Batch number: {batch_number+1}/{len(dataloader)}")
            latents_all = torch.cat(latents_all, dim=0)
            mean = latents_all.mean()
            std = latents_all.std()
            torch.save({"mean": mean, "std": std}, f"outputs/checkpoints/{RUN_NAME}_stats.pt")

            # train the model and save losses
            losses = train_ddpm(
                model=model,
                diffusion=diffusion,
                dataloader=dataloader,
                device=device,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                log_every=LOG_EVERY,
                save_every=SAVE_EVERY,
                run_name= RUN_NAME,
                save_dir= SAVE_DIR,
                timestamp=timestamp,
                latent = True,
                mean = mean,
                std = std,
                autoencoder=autoencoder,
            )
        else:
            print("'TRAINING_ALGORITHM' specified in 'globals.py' is not valid")
    

    # create files for plotting and easy reading
    os.makedirs(LOSSES_DIR, exist_ok=True)
    torch.save(losses, f"{LOSSES_DIR}/{RUN_NAME}_losses_{timestamp}.pt")

    print("\n✓ Losses saved to:")
    print(f"  - {LOSSES_DIR}/{RUN_NAME}_losses_{timestamp}.pt")

    with open(f"{LOSSES_DIR}/{RUN_NAME}_losses_{timestamp}.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    print(f"  - {LOSSES_DIR}/{RUN_NAME}_losses_{timestamp}.txt")
    

if __name__ == "__main__":
    main()