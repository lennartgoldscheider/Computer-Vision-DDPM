import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device

        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )

        return embeddings

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.act = nn.SiLU()

        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        h = self.act(self.conv1(x))

        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[:, :, None, None] # [B, out_channels, 1, 1]

        h = h + time_emb
        h = self.act(self.conv2(h)) # Time Embedding concat and processing

        return h + self.residual(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):

        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.block(x, t)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):

        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_channels * 2, out_channels, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return x
    
class UNet(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=256):

        super().__init__()

        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(64, 128, time_emb_dim)
        self.down2 = DownBlock(128, 256, time_emb_dim)
        self.down3 = DownBlock(256, 512, time_emb_dim)

        self.bottleneck = ConvBlock(512, 512, time_emb_dim)

        self.up1 = UpBlock(512, 512, time_emb_dim)
        self.up2 = UpBlock(512, 256, time_emb_dim)
        self.up3 = UpBlock(256, 128, time_emb_dim)

        self.output_conv = nn.Conv2d(128, image_channels, kernel_size=1)

    def forward(self, x, t):

        t = self.time_mlp(t)
        x = self.input_conv(x)

        x, skip1 = self.down1(x, t)
        x, skip2 = self.down2(x, t)
        x, skip3 = self.down3(x, t)

        x = self.bottleneck(x, t)

        x = self.up1(x, skip3, t)
        x = self.up2(x, skip2, t)
        x = self.up3(x, skip1, t)

        return self.output_conv(x)
