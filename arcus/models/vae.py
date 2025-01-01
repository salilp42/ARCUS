"""Adversarial VAE implementation for outlier detection."""

import torch
import torch.nn as nn


class AdversarialVAE(nn.Module):
    """Adversarial VAE for outlier detection in medical images."""
    
    def __init__(self, input_dim=128, latent_dim=32):
        """Initialize the VAE.
        
        Args:
            input_dim (int): Dimension of input features
            latent_dim (int): Dimension of latent space
        """
        super(AdversarialVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)  # mu and logvar
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim

    def encode(self, x):
        """Encode input to latent space."""
        stats = self.encoder(x)
        mu, logvar = stats[:, :self.latent_dim], stats[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    
    Returns:
        Combined VAE loss
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld
