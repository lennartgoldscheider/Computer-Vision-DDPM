import tarfile
import urllib.request
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from globals import IMAGES_URL, DEST_DIR

# --------------------------------------
# ---------- Download Dataset ----------
# --------------------------------------

# Download Dataset
def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
 
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct // 2)
            print(f"\r  [{'#' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)
 
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
 
 
# unpacks archive and creates useful structure
def extract_flat(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {dest} ...")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for i, member in enumerate(members, 1):
            member.name = Path(member.name).name
            tar.extract(member, path=dest)
            if i % 500 == 0 or i == len(members):
                print(f"  {i}/{len(members)} files", end="\r", flush=True)
    print()


# -----------------------------------------------------
# ---------- Training Dataset and Dataloader ----------
# -----------------------------------------------------

# create Flower Dataset; used for training (with augmentation)
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


# Dataloader for the Flower Dataset for training
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

# -------------------------------------------------------
# ---------- Evaluation Dataset and Dataloader ----------
# -------------------------------------------------------

# create Flower Dataset; used for evaluation (without augmentation)
class ImageFolderDataset(Dataset):
    def __init__(self, root, 
                 image_size=64):
        self.root = Path(root)
        self.paths = list(self.root.glob("*.png")) + list(self.root.glob("*.jpg"))
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor(),  
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
    
# Dataloader for the Flower Dataset for evaluation
def load_images(folder, image_size=64, batch_size=32):
    dataset = ImageFolderDataset(folder, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



# Download Dataset at IMAGES_URL to DEST_DIR by executing this file
def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
 
    archive = Path("/tmp/102flowers.tgz")
    if not archive.exists():
        download(IMAGES_URL, archive)
    else:
        print(f"Archive already cached at {archive}, skipping download.")
 
    extract_flat(archive, DEST_DIR)
 
    images = sorted(DEST_DIR.glob("*.jpg"))
    print(f"\nDone! {len(images)} images in {DEST_DIR}/")
    print(f"Example: {images[0].name} ... {images[-1].name}")
 
 
if __name__ == "__main__":
    main()
