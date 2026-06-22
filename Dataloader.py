from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Dataloader for the Dataset 
# Assumption: all images in same folder & jpg files

class FlowersDataset(Dataset):
    def __init__(self, root, 
                 image_size= 64):
        self.paths = sorted(Path(root).glob("*.jpg")) 
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .jpg files found in {root}")

        # transform images
        self.transform = T.Compose([ 
            T.Resize(image_size),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),               
            T.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]),  
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# Dataloader
def get_dataloader(root,
    image_size= 64,
    batch_size= 64,
    num_workers= 4,
    shuffle = True,
    pin_memory = True,
    drop_last = True):
    dataset = FlowersDataset(root, image_size)
    print(f"Dataset: {len(dataset)} images at {image_size}×{image_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )