import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# This file prepares the dataloader - assuming the dataset is all in the same folder as jpg files

class FlowersDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 64):
        self.paths = sorted(Path(root).glob("*.jpg")) # Collect all jpg's in the given folder
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .jpg files found in {root}")

        self.transform = T.Compose([ # Transforms for diversity and generalization purposes
            T.Resize(image_size),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),               # [0, 1]
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            print("Bad image:", self.paths[idx])
            raise e
        return self.transform(img)


def get_dataloader(
    root: str | Path,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    ) -> DataLoader:
    dataset = FlowersDataset(root, image_size)
    print(f"Dataset: {len(dataset)} images at {image_size}×{image_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )