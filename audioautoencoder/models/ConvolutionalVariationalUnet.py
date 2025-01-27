import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalVariationalUnet(nn.Module):
    def __init__(self, height_resample=1028, width_resample=92, output_height=1025, output_width=89, latent_dim=256):
        super().__init__()

        self.output_height = output_height
        self.output_width = output_width

        self.height_resample = height_resample
        self.width_resample = width_resample
        self.latent_dim = latent_dim

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=1),  # Downsample
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),  # Handles downsampling (may remove stride)
            nn.Dropout2d(p=0.3)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, padding=1),  # Downsample
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),  # Handles downsampling
            nn.Dropout2d(p=0.3)
        )

        # bottleneck
        self.bottleneck_down1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, padding=1),  # Downsample
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Handles downsampling
            nn.Dropout2d(p=0.3)
        )

        # latent space
        self.mu_conv = nn.Conv2d(256, latent_dim, kernel_size=1)  # Mean (μ)
        self.logvar_conv = nn.Conv2d(256, latent_dim, kernel_size=1)  # Log-variance (log(σ²))

        # bottleneck
        self.bottleneck_up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=1, stride=2, padding=1, output_padding=1),  # Downsample
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 64, kernel_size=7, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32 + 16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        # output layer for skip connections
        self.output_layer = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # Final output layer

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random noise ~ N(0, 1)
        return mu + eps * torch.clamp(std, min=1e-6)

    def forward(self, x, verbose=False):
        if verbose:
            print('interpolating', x.shape)
        x = F.interpolate(x, size=(self.height_resample, self.width_resample), mode='bilinear', align_corners=False)
        #print('interpolating', x.shape)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        encoder_output = e3

        # bottleneck
        bn = encoder_output
        bn = self.bottleneck_down1(bn)

        # latent space
        mu = self.mu_conv(bn)
        logvar = self.logvar_conv(bn)
        z = self.reparameterize(mu, logvar)

        # bottleneck
        bn = self.bottleneck_up1(z)


        # Decoder
        decoder_input = F.pad(bn, (0, 0, 0, 0))
        decoder_input = torch.cat((decoder_input, encoder_output), dim=1)
        d3 = self.decoder3(decoder_input) # skip connection

        d3 = F.pad(d3, (0, 1, 0, 1))
        d3 = torch.cat((d3, e2), dim=1)
        d2 = self.decoder2(d3)
        
        d2 = torch.cat((d2, e1), dim=1)
        d1 = self.decoder1(d2)

        output = self.output_layer(d1)
        output = F.interpolate(output, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)

        return output, mu, logvar
