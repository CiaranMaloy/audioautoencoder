import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalUNet(nn.Module):
    def __init__(self, height_resample=1028, width_resample=92, output_height=1025, output_width=89, latent_dim=128):
        super().__init__()

        self.output_height = output_height
        self.output_width = output_width

        self.height_resample = height_resample
        self.width_resample = width_resample
        self.latent_dim = latent_dim

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # Variational Layers (Latent Space)
            # -- remember, height and width may be resampled height and width
        self.fc_mu = nn.Linear(64 * (height_resample // 4) * (width_resample // 4), latent_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(64 * (height_resample // 4) * (width_resample // 4), latent_dim)  # Log variance of latent distribution
        self.fc_decoder = nn.Linear(latent_dim, 64 * (height_resample // 4) * (width_resample // 4))  # Map latent vector back to feature map

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        self.output_layer = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # Final output layer

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random noise ~ N(0, 1)
        return mu + eps * std

    def forward(self, x, verbose=False):
        if verbose:
            print('interpolating', x.shape)
        x = F.interpolate(x, size=(self.height_resample, self.width_resample), mode='bilinear', align_corners=False)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Flatten for latent space
        e3_flat = e3.view(e3.size(0), -1)

        # Compute mean and log variance
        mu = self.fc_mu(e3_flat)
        logvar = self.fc_logvar(e3_flat)

        # Sample from the latent space
        z = self.reparameterize(mu, logvar)

        # Decode latent vector
        z_flat = self.fc_decoder(z).view(e3.size(0), 64, e3.size(2), e3.size(3))

        # Decoder
        d3 = self.decoder3(torch.cat((z_flat, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))

        raw_output = self.output_layer(d1)
        output = F.interpolate(raw_output, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)

        return output, mu, logvar
