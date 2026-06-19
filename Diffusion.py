import torch
import torch.nn.functional as F
import math

# This file is used for adding the noise to the image and removing noise at generation time via sampling

class GaussianDiffusion:

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine",   # "cosine" | "linear"
        device: str = "cpu",):

        self.timesteps = timesteps
        self.device = device

        betas = self._make_schedule(timesteps, schedule)          # (T,)
        alphas = 1.0 - betas                                      # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)             # (T,)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Pre-compute everything the training loop needs. Alpha is the amount of the original image that is remaining.
        # Beta is 1 - alpha. The cummulative product can be used to do the noise addition in one step
        self.register(betas, "betas")
        self.register(alphas, "alphas")
        self.register(alphas_cumprod, "alphas_cumprod")
        self.register(alphas_cumprod_prev, "alphas_cumprod_prev")
        self.register(torch.sqrt(alphas_cumprod), "sqrt_alphas_cumprod")
        self.register(torch.sqrt(1.0 - alphas_cumprod), "sqrt_one_minus_alphas_cumprod")
        self.register(torch.sqrt(1.0 / alphas), "sqrt_recip_alphas")

    # Noise schedule
    def _make_schedule(self, T: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, T)

        if schedule == "cosine":
            steps = T + 1
            t = torch.linspace(0, T, steps) / T
            f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(0, 0.999)

        raise ValueError(f"Unknown schedule: {schedule!r}")

    def register(self, tensor: torch.Tensor, name: str):
        setattr(self, name, tensor.to(self.device))

    # Forward process  q(x_t | x_0)
    def q_sample(self,
        x0: torch.Tensor,           # (B, C, H, W)  clean image in [-1, 1]
        t: torch.Tensor,            # (B,)           integer timesteps
        noise: torch.Tensor = None,) -> tuple[torch.Tensor, torch.Tensor]:

        if noise is None:
            noise = torch.randn_like(x0) # Sample noise

        # for elemenwise multiplication
        sqrt_a = self.sqrt_alphas_prime[t][:, None, None, None]
        sqrt_1a = self.sqrt_one_minus_alphas_prime[t][:, None, None, None]

        x_t = sqrt_a * x0 + sqrt_1a * noise
        return x_t, noise

    # From the T timesteps it chooses a random timestep. This will be used during training.
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)
    
    # Function to reverse a step of noise
    

    # BACKWARD PROCESS ????##

    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = self.betas[t][:, None, None, None] # [B, 1, 1, 1]

        sqrt_one_minus_alpha_bar_t = (self.sqrt_one_minus_alphas_prime[t][:, None, None, None])
        sqrt_recip_alpha_t = (self.sqrt_recip_alphas[t][:, None, None, None])

        predicted_noise = model(x, t)
        model_mean = (sqrt_recip_alpha_t * (x - betas_t * predicted_noise / sqrt_one_minus_alpha_bar_t))

        # last step without noise addition deterministic
        if (t == 0).all():
            return model_mean

        noise = torch.randn_like(x) # Noise addition to make generation flexible
        return model_mean + torch.sqrt(betas_t) * noise
    
    # Sample image from noise - the full pipeline
    @torch.no_grad()
    def sample(self, model, image_size=64, batch_size=16, channels=3):
        model.eval()

        x = torch.randn(batch_size, channels,
            image_size, image_size, device=self.device)

        for timestep in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long) # Timestep for Batch B
            x = self.p_sample(model, x, t)
        print("image done")
        return x