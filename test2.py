from pathlib import Path

from DDPM_generate import generate


CHECKPOINT_DIR = Path("outputs/checkpoints")


def find_checkpoint(epoch: int):

    pattern = f"flowers_ddpm_epoch{epoch}_*.pt"

    matches = sorted(
        CHECKPOINT_DIR.glob(pattern)
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No checkpoint found for epoch {epoch}"
        )

    if len(matches) > 1:
        print(
            f"Warning: multiple checkpoints found "
            f"for epoch {epoch}. Using newest."
        )

    return matches[-1]


def generate_from_epochs(
    epochs,
    num_images=16,
    batch_size=8,
    image_size=64,
    timesteps=1000,
):

    for epoch in epochs:

        checkpoint = find_checkpoint(epoch)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}")
        print(checkpoint.name)
        print("=" * 60)

        generate(
            checkpoint_path=str(checkpoint),
            num_images=num_images,
            batch_size=batch_size,
            image_size=image_size,
            timesteps=timesteps,
        )


def main():

    epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    generate_from_epochs(
        epochs=epochs,
        num_images=16,
        batch_size=8,
        image_size=64,
        timesteps=1000,
    )


if __name__ == "__main__":
    main()