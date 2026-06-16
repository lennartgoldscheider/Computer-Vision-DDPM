import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Autoencoder import Autoencoder
from Dataloader import get_dataloader


# Training

def train_autoencoder(model, dataloader, device,
    epochs=50,
    lr=1e-4,
    log_every=20,
    save_every=10,
    run_name="autoencoder"):

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        epoch_loss = 0.0

        for step, images in enumerate(pbar):
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % log_every == 0:
                losses.append(loss.item())

        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1:03d} | avg loss = {avg_loss:.6f}")

        if ((epoch + 1) % save_every == 0 or (epoch + 1) == epochs):

            checkpoint_path = (checkpoint_dir / f"{run_name}_epoch{epoch+1}_{timestamp}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path)

            print(f"Saved checkpoint:\n {checkpoint_path}")
    return losses

def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Autoencoder(
        in_channels=3,
        latent_channels=4,
        base_channels=64
    ).to(device)

    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=32,
        num_workers=0
    )

    losses = train_autoencoder(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=50,
        save_every=20,
        run_name="flowers_autoencoder"
    )

    torch.save(losses, "training/autoencoder_losses.pt")


if __name__ == "__main__":
    main()