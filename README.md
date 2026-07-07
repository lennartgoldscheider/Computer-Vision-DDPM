# CV-DDPM
Computer Vision project for running DDPM on limited ressources environments
First execute the Load_Dataset file to download the flowers dataset.

Test1 - file to check the Diffusion functionality
Test2 - file to generate images with the given ddpm checkpoints
Test3 - file to test the Autoencoder reconstruction quality
Test4 - file to generate images with the given latent ddpm checkpoints

Load_Dataset    - Download the Flowers Dataset
Dataloader      - Dataloader modularity for the downloaded Flowers Dataset
Diffusion       - Add noise & reverse sampling
Denoising       - Time Embedding and UNet

DDPM_train      - full training pipeline
DDPM_generate   - small generation example with a training checkpoint

Autoencoder     - Autoencoder functionality
Autoencoder_train       - Training of the Autoencoder

latent_DDPM_generate    -
latent_DDPM_train       -

# Computer Vision – Denoising Diffusion Probabilistic Models

This project was developed as part of the Computer Vision class at the Sapienza University in Rome.

The repository contains a complete implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** in **PyTorch**, including data handling, neural network architectures, training, image generation, and quantitative evaluation. The project also explores latent image representations through a convolutional autoencoder and evaluates generated images using **Frechet Inception Distance (FID)** and **Inception Score (IS)**.

---

## Summary

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Authors](#authors)
- [License](#license)

---

## Installation

Clone the repository

```bash
git clone https://github.com/lennartgoldscheider/Computer-Vision-DDPM.git
cd Computer-Vision-DDPM
```

Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install all required packages

```bash
pip install -r requirements.txt
```

---

# Project Structure

The repository is organized into independent modules to simplify experimentation and future extensions.

```
Computer-Vision-DDPM/
│
├── data/             # Dataset download and preprocessing
├── models/           # Autoencoder and DDPM architectures
├── training/         # Training utilities
├── generating/       # Generating utlilities
├── evaluation/       # Evaluation metrics (FID, IS)
├── outputs/          # Checkpoints and generated samples
├── globals/          # Set important parameters for training and generating
├── utils/            # Helper functions
└── README.md
```

---

# Training

The project provides separate modules for

- dataset preparation
- model definition
- training
- image generation
- quantitative evaluation

Model checkpoints are automatically stored inside the output directory.

---

# Evaluation

Generated images can be evaluated using

- Frechet Inception Distance (FID)
- Inception Score (IS)

These metrics provide quantitative comparisons between generated and real images.

---

# Results

Example generated images and reconstruction results can be found in the repository's output folder.

Current experiments investigate the influence of

- latent dimensionality
- autoencoder architecture

on image quality.

---

# Authors

**Lennart Goldscheider**

M.Sc. Machine Learning  
University of Tübingen

GitHub: https://github.com/lennartgoldscheider

---

**Felix Krah**


University 

GitHub: https://github.com/lennartgoldscheider

---

# License

This project is licensed under the MIT License. See the LICENSE file for details.

---
