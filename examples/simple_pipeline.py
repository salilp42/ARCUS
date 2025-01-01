"""Example of using ARCUS for basic image processing."""

import yaml
from pathlib import Path
import torch
from arcus.models.vae import AdversarialVAE
from arcus.models.harmonization import CycleGANHarmonizer
from arcus.core.data_loader import load_and_preprocess_image, validate_site_profiles
from arcus.core.metadata import metadata_verification


def main():
    # Load configuration
    with open("site_profiles.yaml", "r") as f:
        profiles = yaml.safe_load(f)
    validate_site_profiles(profiles)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AdversarialVAE(input_dim=128, latent_dim=32).to(device)
    harmonizer = CycleGANHarmonizer()
    
    # Process a single image
    image_path = "path/to/your/image.dcm"  # Replace with actual path
    img_data = load_and_preprocess_image(image_path)
    
    # Run VAE
    img_tensor = torch.from_numpy(img_data).float().to(device).unsqueeze(0)
    with torch.no_grad():
        recon, mu, logvar = vae(img_tensor)
    
    # Run harmonization
    harmonized = harmonizer.harmonize_image(img_data)
    
    print("Processing complete!")
    

if __name__ == "__main__":
    main()
