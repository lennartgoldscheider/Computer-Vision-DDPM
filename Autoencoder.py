# import torch
# import torch.nn as nn


# # --------------------------------------------------
# # Encoder
# # --------------------------------------------------

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         latent_channels=4,
#         base_channels=64
#     ):
#         super().__init__()

#         self.net = nn.Sequential(

#             # 64 -> 32
#             nn.Conv2d(
#                 in_channels,
#                 base_channels,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1
#             ),
#             nn.SiLU(),

#             # 32 -> 16
#             nn.Conv2d(
#                 base_channels,
#                 base_channels * 2,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1
#             ),
#             nn.SiLU(),

#             # 16 -> 8
#             nn.Conv2d(
#                 base_channels * 2,
#                 base_channels * 4,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1
#             ),
#             nn.SiLU(),

#             nn.Conv2d(
#                 base_channels * 4,
#                 latent_channels,
#                 kernel_size=3,
#                 padding=1
#             )
#         )

#     def forward(self, x):
#         return self.net(x)


# # --------------------------------------------------
# # Decoder
# # --------------------------------------------------

# class Decoder(nn.Module):
#     def __init__(
#         self,
#         latent_channels=4,
#         out_channels=3,
#         base_channels=64
#     ):
#         super().__init__()

#         self.net = nn.Sequential(

#             nn.Conv2d(
#                 latent_channels,
#                 base_channels * 4,
#                 kernel_size=3,
#                 padding=1
#             ),
#             nn.SiLU(),

#             # 8 -> 16
#             nn.ConvTranspose2d(
#                 base_channels * 4,
#                 base_channels * 2,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1
#             ),
#             nn.SiLU(),

#             # 16 -> 32
#             nn.ConvTranspose2d(
#                 base_channels * 2,
#                 base_channels,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1
#             ),
#             nn.SiLU(),

#             # 32 -> 64
#             nn.ConvTranspose2d(
#                 base_channels,
#                 out_channels,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1
#             ),

#             nn.Tanh()
#         )

#     def forward(self, z):
#         return self.net(z)


# # --------------------------------------------------
# # Autoencoder
# # --------------------------------------------------

# class Autoencoder(nn.Module):
#     def __init__(
#         self,
#         in_channels=3,
#         latent_channels=4,
#         base_channels=64
#     ):
#         super().__init__()

#         self.encoder = Encoder(
#             in_channels=in_channels,
#             latent_channels=latent_channels,
#             base_channels=base_channels
#         )

#         self.decoder = Decoder(
#             latent_channels=latent_channels,
#             out_channels=in_channels,
#             base_channels=base_channels
#         )

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):

#         z = self.encode(x)

#         x_hat = self.decode(z)

#         return x_hat

import torch
import torch.nn as nn


# --------------------------------------------------
# Residual Block (NEW)
# --------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.skip(x))


# --------------------------------------------------
# Encoder
# --------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        base_channels=64
    ):
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
        x = self.down3(x)

        x = self.final(x)

        return x


# --------------------------------------------------
# Decoder (FIXED UPSAMPLING)
# --------------------------------------------------

class Decoder(nn.Module):
    def __init__(
        self,
        latent_channels=4,
        out_channels=3,
        base_channels=64
    ):
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
        x = self.up3(x)

        x = torch.tanh(self.out_block(x))

        return x


# --------------------------------------------------
# Autoencoder
# --------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        base_channels=64
    ):
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