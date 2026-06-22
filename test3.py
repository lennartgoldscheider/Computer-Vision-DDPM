import torch
import matplotlib.pyplot as plt
from pathlib import Path
from Autoencoder_fertig import Autoencoder
from Dataloader_fertig import get_dataloader
from utils import load_autoencoder, denormalize

# Reconstruction test of Images by the Autoencoder

# Plot of Original Image vs. Model 1 Reconstruction vs. Model 2 Reconstruction
def save_reconstruction_grid(original, reconstructed1, reconstructed2, title, save_path):

    fig, axes = plt.subplots(1, 3, figsize=(6, 3))

    axes[0].imshow(denormalize(original).permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(denormalize(reconstructed1).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Reconstruction Model 1")
    axes[1].axis("off")

    axes[2].imshow(denormalize(reconstructed2).permute(1, 2, 0).cpu().numpy())
    axes[2].set_title("Reconstruction Model 2")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# test function
def run_test(model1, model2, dataloader, device, save_dir):
    model1.eval()
    model2.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images = next(iter(dataloader)).to(device)

    with torch.no_grad():
        recon1 = model1(images)
        recon2 = model2(images)

    for i in range(8):

        save_reconstruction_grid(
            original=images[i],
            reconstructed1=recon1[i],
            reconstructed2=recon2[i],
            title=f"sample {i}",
            save_path=save_dir / f"sample_{i}.png"
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

    print("\nLoading trained autoencoder...")
    # model_1 = Autoencoder(in_channels=3,
    #     latent_channels=8,
    #     base_channels=64).to(device)
    
    # checkpoint_path_1 = "outputs/checkpoints/flowers_autoencoder_whole_8channels_nownorm_L1_epoch120_20260618_200234.pt" #flowers_autoencoder_3batch_16channels_cont_epoch2000_20260617_173750.pt" #flowers_autoencoder_epoch50_32_20260616_232241.pt" #flowers_autoencoder_epoch50_16_20260616_223642.pt" # flowers_autoencoder_epoch40_16_20260616_223642.pt" # flowers_autoencoder_epoch20_16_20260616_223642.pt" # flowers_autoencoder_epoch50_20260616_220930.pt"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model_1.load_state_dict(checkpoint["model_state_dict"])
    
    # model_2 = Autoencoder(in_channels=3,
    #     latent_channels=16,
    #     base_channels=64).to(device)
    
    # checkpoint_path_2 = "outputs/checkpoints/flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt" #flowers_autoencoder_whole_16channel_epoch160_20260617_185844.pt" #flowers_autoencoder_3batch_32channel_epoch2000_20260617_182211.pt"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model_2.load_state_dict(checkpoint["model_state_dict"])

    checkpoint_path_1 = "outputs/checkpoints/flowers_autoencoder_whole_8channels_nownorm_L1_epoch120_20260618_200234.pt" 
    checkpoint_path_2 = "outputs/checkpoints/flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt"    
    
    model_1 = load_autoencoder(checkpoint_path_1, device, 
                     latent_channels=8)
    
    model_2 = load_autoencoder(checkpoint_path_2, device, 
                     latent_channels=16)
    

    run_test(
        model_1,
        model_2,
        dataloader,
        device,
        save_dir="outputs/autoencoder_test"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()