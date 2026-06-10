"""
Download Oxford 102 Flowers dataset.
All images are placed flat in datasets/flowers/ (no subfolders, no labels).
"""
 
import os
import tarfile
import urllib.request
from pathlib import Path
 
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
DEST_DIR = Path("datasets/flowers")
 
 
def download(url: str, dest: Path) -> None:
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
 
 
def extract_flat(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {dest} ...")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for i, member in enumerate(members, 1):
            member.name = Path(member.name).name  # strip any subdirectory
            tar.extract(member, path=dest)
            if i % 500 == 0 or i == len(members):
                print(f"  {i}/{len(members)} files", end="\r", flush=True)
    print()
 
 
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
