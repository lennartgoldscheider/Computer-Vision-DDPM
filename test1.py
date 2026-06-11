"""
Schnelltest für dataset.py und diffusion.py.

Erstellt ein synthetisches Blumenbild (falls kein echter Datensatz vorhanden),
führt den Forward-Diffusion-Prozess durch und speichert eine Vergleichsgrafik.

Ausführen:
    python test_pipeline.py
    python test_pipeline.py --data datasets/flowers   # echter Datensatz
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# Eigene Module
sys.path.insert(0, str(Path(__file__).parent))
from Dataloader import FlowersDataset, get_dataloader
from Diffusion import GaussianDiffusion


# ── Hilfsfunktionen ────────────────────────────────────────────────────────────

def make_dummy_dataset(tmp_dir: Path, n: int = 8, size: int = 128):
    """Erstellt synthetische Blumenbilder (Kreise + Farbe) für den Test."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        bg = int(rng.integers(180, 240))
        img = np.full((size, size, 3), bg, dtype=np.uint8)
        # Einfacher Kreis als "Blume"
        cx, cy = size // 2, size // 2
        r = size // 3
        ys, xs = np.ogrid[:size, :size]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        color = rng.integers(100, 255, 3, dtype=np.uint8)
        img[mask] = color
        Image.fromarray(img).save(tmp_dir / f"image_{i:05d}.jpg")
    return tmp_dir


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Tensor (C,H,W) in [-1,1] → numpy (H,W,3) in [0,1]."""
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


# ── Haupttest ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/flowers",
                        help="Pfad zum Flowers-Ordner (optional)")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--out", type=str, default="test_noise_output.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if args.data and Path(args.data).exists():
        data_dir = Path(args.data)
        print(f"Echter Datensatz: {data_dir}")
    else:
        data_dir = Path("/tmp/flowers_dummy")
        print("Kein Datensatz angegeben → erstelle Dummy-Bilder ...")
        make_dummy_dataset(data_dir, n=16, size=max(args.image_size * 2, 128))

    loader = get_dataloader(
        root=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,  # für den Test reicht 0
    )
    batch = next(iter(loader))          # (B, 3, H, W)
    print(f"Batch shape: {batch.shape}  min={batch.min():.2f}  max={batch.max():.2f}")
    assert batch.min() >= -1.1 and batch.max() <= 1.1, "Normalisierung außerhalb [-1,1]!"
    print("✓ Dataset & Normalisierung OK")

    # ── 2. Diffusion ──────────────────────────────────────────────────────────
    diff = GaussianDiffusion(timesteps=args.timesteps, schedule="cosine", device=device)
    x0 = batch[:1].to(device)          # erstes Bild aus dem Batch

    timesteps_to_show = [0, 100, 250, 500, 750, 999]
    noisy_images = []
    for t_val in timesteps_to_show:
        t = torch.tensor([t_val], device=device)
        x_t, _ = diff.q_sample(x0, t)
        noisy_images.append(x_t.squeeze(0))

    print(f"✓ q_sample für {len(timesteps_to_show)} Zeitschritte OK")

    # Stichprobe: zufällige Timesteps für einen Batch
    t_rand = diff.sample_timesteps(args.batch_size)
    x_batch = batch.to(device)
    x_noisy, noise = diff.q_sample(x_batch, t_rand)
    assert x_noisy.shape == x_batch.shape, "Shape mismatch nach q_sample!"
    print(f"✓ Batch q_sample OK  (t range: {t_rand.min()}–{t_rand.max()})")

    # ── 3. Grafik speichern ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 3.5), facecolor="#1a1a1a")
    gs = gridspec.GridSpec(1, len(timesteps_to_show) + 1, figure=fig,
                           wspace=0.08, left=0.02, right=0.98,
                           top=0.82, bottom=0.05)

    # Original
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(tensor_to_numpy(x0.squeeze(0)))
    ax0.set_title("original", color="white", fontsize=9, pad=4)
    ax0.axis("off")
    ax0.set_facecolor("#1a1a1a")

    # Noisy steps
    for i, (t_val, img_t) in enumerate(zip(timesteps_to_show, noisy_images)):
        ax = fig.add_subplot(gs[i + 1])
        ax.imshow(tensor_to_numpy(img_t))
        ax.set_title(f"t={t_val}", color="white", fontsize=9, pad=4)
        ax.axis("off")
        ax.set_facecolor("#1a1a1a")

    fig.suptitle("Forward Diffusion  q(x_t | x_0)  — Cosine Schedule",
                 color="white", fontsize=11, y=0.97)

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n✓ Grafik gespeichert → {out_path.resolve()}")
    print("\nAlle Tests bestanden ✓")


if __name__ == "__main__":
    main()