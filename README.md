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

This project was developed as part of my work in deep generative models and computer vision during my M.Sc. in Machine Learning at the University of Tübingen.

The repository contains a complete implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** in **PyTorch**, including data handling, neural network architectures, training, image generation, and quantitative evaluation. The project also explores latent image representations through a convolutional autoencoder and evaluates generated images using **Frechet Inception Distance (FID)** and **Inception Score (IS)**.

---

## Summary

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

# Getting Started

These instructions describe how to set up the project locally for training and evaluation.

## Prerequisites

- Ubuntu 22.04 (tested)
- Python 3.12+
- PyTorch
- CUDA-compatible GPU (recommended)
- Git

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
├── evaluation/       # Evaluation metrics (FID, IS)
├── outputs/          # Checkpoints and generated samples
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
- reconstruction loss functions
- diffusion hyperparameters

on image quality.

---

# Future Improvements

Potential future extensions include

- DDIM sampling
- Classifier-Free Guidance
- Exponential Moving Average (EMA)
- Attention blocks
- Conditional diffusion models
- Latent Diffusion Models (LDM)

---

# Authors

**Lennart Goldscheider**

M.Sc. Machine Learning  
University of Tübingen

GitHub: https://github.com/lennartgoldscheider

---

# License

This project is licensed under the MIT License. See the LICENSE file for details.

---

# Acknowledgments

- Jonathan Ho et al. for introducing Denoising Diffusion Probabilistic Models.
- The PyTorch team for the deep learning framework.
- The University of Tübingen for supporting this project.
