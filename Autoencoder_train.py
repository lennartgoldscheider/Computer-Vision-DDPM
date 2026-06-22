import time
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Autoencoder_fertig import Autoencoder
from Dataloader_fertig import get_dataloader

# Training of the autoencoder

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

        # compute average loss per epoch
        avg_loss = epoch_loss / len(dataloader)

        # print in between loss
        print(f"Epoch {epoch+1:03d} | avg loss = {avg_loss:.6f}")

        # save model
        if (epoch + 1) % save_every == 0:
            model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)

    # final save
    model.save_checkpoint(epoch, save_dir, run_name, timestamp, optimizer, avg_loss)

    return losses

def main():

    # set parameters
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_dir = "outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_name = "flowers_autoencoder_whole_8channels_norm"
    timestamp = time.strftime("%Y%m%d_%H%M")

    # set models and dataloader
    model = Autoencoder(
        in_channels=3,
        latent_channels=8,
        base_channels=64
    ).to(device)
    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=32,
        num_workers=0
    )

    # train the model and save losses
    losses = train_autoencoder(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=2000,
        save_every=1,#20,
        run_name=run_name,
        timestamp= timestamp
    )

    # create files for plotting and easy reading
    losses_save_dir = "outputs/training"
    os.makedirs(losses_save_dir, exist_ok=True)
    torch.save(losses, f"outputs/training/autoencoder_losses_{timestamp}.pt")

    print("\n✓ Losses saved to:")
    print(f"  - output/training/autoencoder_losses_{timestamp}.pt")

    loss_path_txt = os.path.join(losses_save_dir, f"autoencoder_losses_{timestamp}.txt")
    with open(loss_path_txt, "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")
    print(f"  - {loss_path_txt}")


if __name__ == "__main__":
    main()
