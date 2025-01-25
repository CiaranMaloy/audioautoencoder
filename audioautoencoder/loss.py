# custom loss function
import torch
import torch.nn as nn
import torch.nn.functional as F

class MelWeightedMSELoss(nn.Module):
    def __init__(self, device, height=1025, min_value=0.3):
        super(MelWeightedMSELoss, self).__init__()

        self.height = height
        self.min_value = min_value

        f = (torch.arange(height, device=device).float().unsqueeze(1) + 20) * 20000

        # Compute the mel scale using the formula
        mel = 1127.01048 * torch.log(f / 700 + 1)

        # Normalize mel and compute the inverse
        n_mel = 1 - (mel / torch.max(mel))

        # Calculate weights
        self.weights = (n_mel + self.min_value) / torch.max(n_mel + self.min_value)

    def forward(self, input, target):
        self.weights.to(input.device)

        loss = ((input - target) ** 2) * self.weights
        return torch.mean(loss)
    
class MelWeightedMSELossVAE(nn.Module):
    def __init__(self, device, height=1025, min_value=0.3, verbose=False, beta=0.1):
        super(MelWeightedMSELossVAE, self).__init__()
        self.verbose = verbose
        if self.verbose:
            print('loss initialised...')

        self.beta = beta

        self.height = height
        self.min_value = min_value

        f = (torch.arange(height, device=device).float().unsqueeze(1) + 20) * 20000

        # Compute the mel scale using the formula
        mel = 1127.01048 * torch.log(f / 700 + 1)

        # Normalize mel and compute the inverse
        n_mel = 1 - (mel / torch.max(mel))

        # Calculate weights
        self.weights = (n_mel + self.min_value) / torch.max(n_mel + self.min_value)

    def forward(self, arguments, target, beta=None):
        if beta is not None:
            self.beta = beta
        # extract arguments
        input, mu, logvar = arguments
        logvar = torch.clamp(logvar, min=-10, max=10)
        # Ensure weights are on the same device as input
        weights = self.weights.to(input.device)
        # perceptual loss
        recon_loss = torch.mean(((input - target) ** 2) * weights)

        # kl loss 
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        kl_loss /= input.size(0)
        kl_loss = torch.clamp(kl_loss, min=1e-6) 

        if self.verbose:
            print('kl loss:', kl_loss)
            print('recon_loss:', recon_loss)

        return recon_loss + kl_loss * self.beta, recon_loss
    
def vae_loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (e.g., MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss