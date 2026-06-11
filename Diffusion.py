import torch
import torch.nn.functional as F
import math


class GaussianDiffusion:
    """
    Forward diffusion process (noise addition only).
    Reverse (sampling) comes later in train.py / sample.py.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",   # "cosine" | "linear"
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = device

        betas = self._make_schedule(timesteps, schedule)          # (T,)
        alphas = 1.0 - betas                                      # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)             # (T,)

        # Pre-compute everything the training loop needs
        self.register(betas, "betas")
        self.register(alphas_cumprod, "alphas_cumprod")
        self.register(torch.sqrt(alphas_cumprod), "sqrt_alphas_cumprod")
        self.register(torch.sqrt(1.0 - alphas_cumprod), "sqrt_one_minus_alphas_cumprod")

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def _make_schedule(self, T: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, T)

        if schedule == "cosine":
            # Nichol & Dhariwal 2021 cosine schedule
            steps = T + 1
            t = torch.linspace(0, T, steps) / T
            f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(0, 0.999)

        raise ValueError(f"Unknown schedule: {schedule!r}")

    def register(self, tensor: torch.Tensor, name: str):
        setattr(self, name, tensor.to(self.device))

    # ------------------------------------------------------------------
    # Forward process  q(x_t | x_0)
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,           # (B, C, H, W)  clean image in [-1, 1]
        t: torch.Tensor,            # (B,)           integer timesteps
        noise: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to x0 at timestep t.
        Returns (noisy_image, noise) so the training loop can supervise on noise.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_a * x0 + sqrt_1a * noise
        return x_t, noise

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Uniform random timesteps for a training batch."""
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)