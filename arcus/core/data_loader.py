"""Data loading and preprocessing utilities."""

import numpy as np
import pydicom
import nibabel as nib
from pathlib import Path
from typing import Union, Dict, Any


def load_and_preprocess_image(file_path: Union[str, Path]) -> np.ndarray:
    """Load and preprocess medical image data.
    
    Supports DICOM and NIfTI formats.
    
    Args:
        file_path: Path to image file
        
    Returns:
        Preprocessed image data as flattened array
    """
    file_path = str(file_path)
    
    if file_path.endswith(".dcm"):
        ds = pydicom.dcmread(file_path)
        img_data = ds.pixel_array.astype(np.float32)
    else:
        img = nib.load(file_path)
        img_data = img.get_fdata().astype(np.float32)

    img_flat = img_data.ravel()
    
    # Ensure exactly 128 elements
    if img_flat.shape[0] > 128:
        img_flat = img_flat[:128]
    else:
        pad_len = 128 - img_flat.shape[0]
        img_flat = np.pad(img_flat, (0, pad_len), 'constant', constant_values=0)
        
    return img_flat


def validate_site_profiles(profiles: Dict[str, Any]) -> None:
    """Validate site profile configuration.
    
    Args:
        profiles: Dictionary containing site profiles
        
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ['site_name', 'scanner_info', 'protocol_ranges']
    for site in profiles:
        missing = [k for k in required_keys if k not in profiles[site]]
        if missing:
            raise ValueError(f"Site '{site}' missing required keys: {missing}")
