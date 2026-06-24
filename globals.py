from pathlib import Path

# ---------- data.py ----------
IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
DEST_DIR = Path("datasets/flowers")



# -------- training.py --------
AUTOENCODER_TRAINING_CHECKPOINT = (
            "outputs/checkpoints/"
            "flowers_autoencoder_whole_4channels_nownorm_L1_epoch240_20260618_214733.pt"
            )
LOSSES_DIR = "outputs/training"
SAVE_DIR = "outputs/checkpoints"

TRAINING_ALGORITHM = "Latent DDPM" #"DDPM" #"Autoencoder" # "DDPM" "Latent DDPM"

TIMESTEPS_TRAINING = 1000        # Timesteps for sampling
SCHEDULE_TRAINING = "cosine"     # Beta schedule for sampling
LATENT_CHANNELS_TRAINING = 4     
IMAGE_SIZE_TRAINING = 64
BATCH_SIZE_TRAINING = 32
BASE_CHANNELS_TRAINING = 64
EPOCHS = 200
SAVE_EVERY = 20                  # Save Checkpoint every ... epochs
LOG_EVERY = 20                   # Save losses every ... epochs
LEARNING_RATE = 1e-4

name_for_saving = "autoencoder" if TRAINING_ALGORITHM == "Autoencoder" else "ddpm" if TRAINING_ALGORITHM == "DDPM" else "latent_ddpm"
RUN_NAME = f"flowers_{name_for_saving}_{LATENT_CHANNELS_TRAINING}channels"


# -------- generating.py --------
LATENT_DDPM_CHECKPOINT = (
            "outputs/checkpoints/"
            "flowers_latent_ddpm_16channels_epoch100_20260618_175522.pt"
            )
AUTOENCODER_GENERATING_CHECKPOINT = (
            "outputs/checkpoints/"
            "flowers_autoencoder_whole_16channels_nownorm_L1_epoch200_20260618_113626.pt"
            )
DDPM_CHECKPOINT = ("outputs/checkpoints/"
            "flowers_ddpm_epoch1_20260618_1640.pt"
            )
OUTPUT_ROOT = "outputs/samples"

GENERATING_ALGORITHM = "Latent DDPM" #"Autoencoder" # "DDPM" "Latent DDPM"

NUM_IMAGES_GENERATING = 4#64           # if None then autoencoder reconstructs whole dataset
TIMESTEPS_GENERATING = 1000        # Timesteps for sampling
SCHEDULE_GENERATING = "cosine"     # Beta schedule for sampling
BATCH_SIZE_GENERATING = 8 
LATENT_SIZE = 16
LATENT_CHANNELS_GENERATING = 16
IMAGE_SIZE_GENERATING = 64


# -------- evaluation.py --------
REAL_DIR = "datasets/flowers"

# calculate FID and IS for each of these directories (of samples) 
LIST_SAMPLES_DIR = ["outputs/samples/flowers_latent_ddpm_16channels",
                    ]
