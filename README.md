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
