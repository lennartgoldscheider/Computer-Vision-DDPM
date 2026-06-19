import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Denoising import UNet
from Diffusion import GaussianDiffusion
from Dataloader import get_dataloader

# Training pipeline for DDPM.
def train(model, diffusion, dataloader, device,
    epochs=1,
    lr=1e-4,
    log_every=10,
    save_dir="outputs/checkpoints",
    run_name="ddpm",
    save_every_epochs=1,
    timestamp = 0):

    # def save_checkpoint(epoch):
    #     path = os.path.join(save_dir, f"{run_name}_epoch{epoch+1}_{timestamp}.pt")
    #     torch.save({
    #         "epoch": epoch,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "losses": avg_loss,
    #         "timestamp": timestamp
    #     }, path)
    #     print(f"\n✓ checkpoint saved → {path}")


    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    pbar = tqdm(total=epochs * len(dataloader)) # progress bar

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, x_0 in enumerate(dataloader):

            # get x0, t and noise
            x_0 = x_0.to(device)
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

        # compute average loss per epoch
        avg_loss = epoch_loss / len(dataloader)

        # save model
        if (epoch + 1) % save_every_epochs == 0:
            model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)
            # save_checkpoint(epoch)

    # final save
    model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)
    #save_checkpoint(epochs-1)

    return losses


def main():

    # set parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    save_dir = "outputs/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    run_name = "flowers_ddpm"
    timestamp = time.strftime("%Y%m%d_%H%M")

    # set models and dataloader
    model = UNet()
    diffusion = GaussianDiffusion(
        timesteps=1000,
        schedule="cosine",
        device=device
    )
    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=32,
        num_workers=0
    )

    # train the model and save losses
    losses = train(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        device=device,
        epochs=200,
        log_every=100,
        run_name=run_name,
        save_dir=save_dir,
        save_every_epochs=1,
        timestamp = timestamp,
    )

    # create files for plotting and easy reading
    loss_path_npy = os.path.join(save_dir, f"{run_name}_losses_{timestamp}.npy")
    np.save(loss_path_npy, np.array(losses))

    print("\n✓ Losses saved to:")
    print(f"  - {loss_path_npy}")

    loss_path_txt = os.path.join(save_dir, f"{run_name}_losses_{timestamp}.txt")
    with open(loss_path_txt, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    print(f"  - {loss_path_txt}")

if __name__ == "__main__":
    main()
