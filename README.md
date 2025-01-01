# ARCUS: Automated Radiological Curation & Unification System

ARCUS is a comprehensive pipeline for curating and harmonizing multi-site medical imaging datasets. It implements state-of-the-art techniques for data quality control, outlier detection, and domain adaptation, with optional federated learning capabilities for privacy-preserving multi-center collaboration.

## Features

### Core Functionality
- Site-specific quality control and curation
- Advanced outlier detection using Adversarial VAE
- Domain adaptation via CycleGAN-based harmonization
- LLM-based metadata verification
- Automated reporting and visualization

### Federated Learning Extensions
- Privacy-preserving distributed training
- Differential privacy integration
- Smart client selection strategies
- Robust model evaluation
- Checkpoint management

## Installation

```bash
git clone https://github.com/salilpatel/ARCUS.git
cd ARCUS
pip install -r requirements.txt
```

## Quick Start

### Basic Usage (Single Site)
```bash
python examples/simple_pipeline.py \
    --data_dir /path/to/multisite_data \
    --site_profiles examples/site_profiles.yaml \
    --output_dir /path/to/curated_data
```

### Federated Mode
```bash
python examples/federated_pipeline.py \
    --federated True \
    --data_dir /path/to/multisite_data \
    --site_profiles examples/site_profiles.yaml \
    --output_dir /path/to/curated_data \
    --min_site_samples 50 \
    --rounds 5 \
    --local_epochs 2
```

## Project Structure
```
arcus/
├── core/              # Core functionality
│   ├── data_loader.py # Data loading utilities
│   └── metadata.py    # Metadata verification
├── federated/         # Federated learning components
│   └── config.py      # Federated config
├── models/            # Neural network models
│   ├── vae.py        # Adversarial VAE
│   └── harmonization.py # Image harmonization
└── utils/            # Helper utilities
```

## Configuration

### Site Profiles (YAML)
```yaml
site_a:
  site_name: "Hospital A"
  scanner_info:
    manufacturer: "Siemens"
    model: "Skyra"
  protocol_ranges:
    slice_thickness: [1.0, 3.0]
    tr_range: [1500, 2500]
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License - see LICENSE file for details.

## Citation
If using ARCUS in research, please cite:
```bibtex
@software{patel2024arcus,
  author = {Patel, Salil},
  title = {ARCUS: Automated Radiological Curation \& Unification System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/salilpatel/ARCUS}
}
```
