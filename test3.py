import torch
import matplotlib.pyplot as plt
from pathlib import Path

from Autoencoder import Autoencoder
from Dataloader import get_dataloader

# Generation test of Autoencoder Quality


def tensor_to_img(x):
    # [-1,1] tensor -> [0,1] numpy image (H,W,C)
    x = (x.clamp(-1, 1) + 1) / 2
    return x.permute(1, 2, 0).cpu().numpy()


def save_reconstruction_grid(original, reconstructed, title, save_path):

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(tensor_to_img(original))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_img(reconstructed))
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# test function
def run_test(model, dataloader, device, tag, save_dir):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images = next(iter(dataloader)).to(device)

    with torch.no_grad():
        recon = model(images)

    for i in range(8):  # save first 8 examples

        save_reconstruction_grid(
            original=images[i],
            reconstructed=recon[i],
            title=f"{tag} sample {i}",
            save_path=save_dir / f"{tag}_sample_{i}.png"
        )
    print(f"Saved reconstructions to {save_dir}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(
        root="datasets/flowers",
        image_size=64,
        batch_size=8,
        num_workers=0
    )

    # Run test for both untrained version (baseline) and trained version
    print("\nTesting untrained autoencoder...")
    model_untrained = Autoencoder().to(device)

    run_test(
        model_untrained,
        dataloader,
        device,
        tag="untrained",
        save_dir="outputs/autoencoder_test/untrained"
    )

    print("\nLoading trained autoencoder...")
    model_trained = Autoencoder().to(device)

    checkpoint_path = "outputs/checkpoints/flowers_autoencoder_epoch50_20260614_171314.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_trained.load_state_dict(checkpoint["model_state_dict"])

    run_test(
        model_trained,
        dataloader,
        device,
        tag="trained",
        save_dir="outputs/autoencoder_test/trained"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()