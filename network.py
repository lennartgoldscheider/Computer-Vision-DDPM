import torch
import torch.nn as nn
import os
import math

# ------------------------------------------------
# ---------- Autoencoder Architecture   ----------
# ---------- latent space approximation ----------
# ------------------------------------------------

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        groups = min(8, in_ch)
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        h = self.act(self.conv1(self.norm1(x)))
        h = self.conv2(self.norm2(h))
        return self.act(h + self.skip(x))
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, base_channels=64):
        super().__init__()

        self.block1 = ResBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.block2 = ResBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.block3 = ResBlock(base_channels * 4, base_channels * 4)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        self.final = nn.Conv2d(base_channels * 4, latent_channels, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.down1(x)

        x = self.block2(x)
        x = self.down2(x)

        x = self.block3(x)

        x = self.final(x)

        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3, base_channels=64):
        super().__init__()

        self.init = nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1)
        self.block1 = ResBlock(base_channels * 4, base_channels * 4)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)
        )

        self.block2 = ResBlock(base_channels * 4, base_channels * 2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        )

        self.block3 = ResBlock(base_channels * 2, base_channels)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, base_channels, 3, padding=1)
        )

        self.out_block = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, z):
        x = self.act(self.init(z))

        x = self.block1(x)
        x = self.up1(x)

        x = self.block2(x)
        x = self.up2(x)

        x = self.block3(x)

        x = torch.tanh(self.out_block(x))

        return x


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, base_channels=64):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels
        )

        self.decoder = Decoder(
            latent_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def save_checkpoint(self,epoch, save_dir, run_name, timestamp, optimizer, avg_loss):
        path = os.path.join(save_dir, f"{run_name}_epoch{epoch+1}_{timestamp}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": avg_loss,
            "timestamp": timestamp
        }, path)
        print(f"\n✓ checkpoint saved → {path}")


# -----------------------------------------
# ---------- UNet Architecture   ----------
# ---------- for Denoising       ----------
# -----------------------------------------

# Converts scalar t into a meaningfull vector
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

# Convolution Layer with added time embedding to increase receptive field 
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
        time_emb = time_emb[:, :, None, None] # Broadcast

        h = h + time_emb
        h = self.act(self.conv2(h)) # Time Embedding concat and processing

        return h + self.residual(x)

# Resolution reduction, channel / feature information increase
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

# Resolution increase, channel / feature information decrease and
# Concatination with skip information.
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

# Combine the various features for the full UNet
class UNet(nn.Module):
    def __init__(self, 
                 image_channels=3, 
                 base_channels=64, 
                 time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels*2, time_emb_dim)
        self.down2 = DownBlock(base_channels*2, base_channels*4, time_emb_dim)
        self.down3 = DownBlock(base_channels*4, base_channels*8, time_emb_dim)

        self.bottleneck = ConvBlock(base_channels*8, base_channels*8, time_emb_dim)

        self.up1 = UpBlock(base_channels*8, base_channels*8, time_emb_dim)
        self.up2 = UpBlock(base_channels*8, base_channels*4, time_emb_dim)
        self.up3 = UpBlock(base_channels*4, base_channels*2, time_emb_dim)

        self.output_conv = nn.Conv2d(base_channels*2, image_channels, kernel_size=1)

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
    
    def save_checkpoint(self,epoch, save_dir, run_name, timestamp, optimizer, avg_loss):
        path = os.path.join(save_dir, f"{run_name}_epoch{epoch+1}_{timestamp}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": avg_loss,
            "timestamp": timestamp
        }, path)
        print(f"\n✓ checkpoint saved → {path}")