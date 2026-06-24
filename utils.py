import torch
import math
from network import UNet, Autoencoder

# Convert [-1,1] to [0,1]
def denormalize(x):
    return ((x.clamp(-1, 1) + 1) / 2)

# Load Latent DDPM
def load_diffusion_model(checkpoint_path, device,
                         image_channels=3,
                         base_channels=64,
                         time_emb_dim=256):

    # initialize UNet
    model = UNet(
        image_channels=image_channels,
        base_channels=base_channels,
        time_emb_dim=time_emb_dim,
    ).to(device)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# Load Autoencoder
def load_autoencoder(checkpoint_path, device, 
                     latent_channels=4,
                     base_channels=64):

    # initialize Autoencoder
    autoencoder = Autoencoder(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    # load Autoencoder
    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    return autoencoder

def freeze_autoencoder(autoencoder):
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False


# Diffusion process class
class GaussianDiffusion:
    def __init__(self, 
                 timesteps = 1000, 
                 schedule = "cosine",   # "cosine" or "linear"
                 device = "cpu",):

        self.timesteps = timesteps
        self.device = device

        # setting parameters for Diffusion
        betas = self._make_schedule(timesteps, schedule)          
        alphas = 1.0 - betas                                      
        alphas_prime = torch.cumprod(alphas, dim=0)             
        alphas_prime_prev = torch.cat([torch.tensor([1.0]), alphas_prime[:-1]])

        # Pre-compute everything the training loop needs (according to the lecture)
        self.register(betas, "betas")
        self.register(alphas, "alphas")
        self.register(alphas_prime, "alphas_prime")
        self.register(alphas_prime_prev, "alphas_prime_prev")
        self.register(torch.sqrt(alphas_prime), "sqrt_alphas_prime")
        self.register(torch.sqrt(1.0 - alphas_prime), "sqrt_one_minus_alphas_prime")
        self.register(torch.sqrt(1.0 / alphas), "sqrt_recip_alphas")

    # Noise schedule
    def _make_schedule(self, T, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, T)

        if schedule == "cosine":
            steps = T + 1
            t = torch.linspace(0, T, steps) / T
            f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

            return betas.clamp(0, 0.4) #betas.clamp(0, 0.999)

        raise ValueError(f"Unknown schedule: {schedule!r}")

    def register(self, tensor: torch.Tensor, name: str):
        setattr(self, name, tensor.to(self.device))

    # Forward process  q(x_t | x_0)
    def q_sample(self,x_0, t,
        noise = None,):
        # x_0 is in  [-1,1]

        # create noise if necessary
        if noise is None:
            noise = torch.randn_like(x_0)

        # Broadcasting
        sqrt_a = self.sqrt_alphas_prime[t][:, None, None, None]
        sqrt_1a = self.sqrt_one_minus_alphas_prime[t][:, None, None, None]

        # compute x_t
        x_t = sqrt_a * x_0 + sqrt_1a * noise

        return x_t, noise

    # sample random timesteps for the whole batch
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)
    
    # Backward process q(x_t-1|x_t,x_0)
    @torch.no_grad()
    def p_sample(self, model, x, t):

        # Broadcasting
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = (self.sqrt_one_minus_alphas_prime[t][:, None, None, None])
        sqrt_recip_alpha_t = (self.sqrt_recip_alphas[t][:, None, None, None])

        # predict the noise
        predicted_noise = model(x, t)

        # compute the mean mu_t
        model_mean = (sqrt_recip_alpha_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_bar_t))

        # last step without noise addition -> deterministic
        # otherwise return x_t-1
        if (t == 0).all():
            return model_mean
        else:
            return model_mean + torch.sqrt(betas_t) * torch.randn_like(x)
    
    # Sampling algorithm (noise to image)
    @torch.no_grad()
    def sample(self, model, 
               image_size=64, 
               batch_size=16, 
               channels=3):
        
        model.eval()

        # create random noise images
        x = torch.randn(batch_size, channels,
            image_size, image_size, device=self.device)
        
        # Backward loop
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        print("image done")
        return x
