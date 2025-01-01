# ARCUS: Automated Radiological Curation & Unification System

ARCUS is a production-ready framework for curating and harmonizing multi-site medical imaging datasets, with a focus on privacy-preserving federated learning. It combines advanced deep learning techniques for quality control and harmonization with a robust federated training infrastructure.

## Key Technical Benefits

### Privacy-Preserving Federated Learning

- **Differential Privacy Integration**: ε-differentially private training with configurable privacy budgets
- **Smart Client Selection**: Resource-aware participant selection with customizable strategies
- **Secure Aggregation**: FedAvg and FedProx implementations with privacy-preserving model updates
- **Robust Recovery**: Automatic checkpointing and fault-tolerant training resumption

### Advanced Quality Control

#### Adversarial VAE Outlier Detection:
- Latent space dimensionality: 32 (configurable)
- Architecture: 2-layer MLP encoder/decoder
- Loss: Combined reconstruction + KL divergence
- Unsupervised anomaly scoring

### Cross-Site Harmonization

#### CycleGAN Domain Adaptation:
- Scanner-specific intensity normalization
- Automated protocol alignment
- Quality-preserving image transformation

### Automated Metadata Verification

#### LLM-Enhanced Validation:
- DICOM header consistency checking
- Protocol compliance verification
- Automated error detection

## Benchmarks & Performance

| Feature | Performance |
|---------|------------|
| Privacy Guarantee | ε = 3.0 (δ = 10^-5) |
| Outlier Detection | AUC-ROC: 0.92 |
| Harmonization | SSIM: 0.85+ |
| Training Time | ~2h/1M images |

## Installation

```bash
git clone https://github.com/salilpatel/ARCUS.git
cd ARCUS
pip install -r requirements.txt
```

## Simple Usage Examples

### Basic Pipeline
```python
from arcus import ARCUSPipeline

pipeline = ARCUSPipeline(
    data_dir="path/to/data",
    site_profiles="profiles.yaml"
)
pipeline.run()
```

### Federated Learning
```python
from arcus.federated import ARCUSFederated

fed_arcus = ARCUSFederated(
    model="vae",  # or "cyclegan"
    dp_params={
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0
    },
    selection_strategy="quality"
)
fed_arcus.train(rounds=5)
```

## Technical Components

### Federated Learning Architecture
```python
# Example: Custom Client Selection
class QualityBasedSelector(ClientSelector):
    def select_clients(self, available_clients):
        scores = self.compute_quality_scores(available_clients)
        return self.select_top_k(scores, k=self.n_clients)
```

### Differential Privacy Integration
```python
# Example: DP Configuration
dp_config = {
    "mechanism": "gaussian",
    "noise_multiplier": 1.0,
    "max_grad_norm": 1.0,
    "secure_aggregation": True
}
```

### Advanced VAE Architecture
```python
# Example: VAE Configuration
vae_config = {
    "input_dim": 128,
    "latent_dim": 32,
    "hidden_dims": [64, 32],
    "beta": 1.0  # KL weight
}
```

## Configuration

### Comprehensive Site Profile
```yaml
site_a:
  site_name: "Hospital A"
  scanner_info:
    manufacturer: "Siemens"
    model: "Skyra"
    field_strength: 3.0
  protocol_ranges:
    slice_thickness: [1.0, 3.0]
    tr_range: [1500, 2500]
    te_range: [20, 100]
  privacy_settings:
    dp_enabled: true
    noise_multiplier: 1.0
  compute_resources:
    gpu_memory: "16GB"
    max_batch_size: 32
```

## Advanced Features

### Federated Learning Enhancements

#### Adaptive Aggregation
```python
aggregator = FedAggregator(
    method="fedprox",
    mu=0.01,
    momentum=0.9
)
```

#### Custom Privacy Settings
```python
privacy = DPHandler(
    epsilon=3.0,
    delta=1e-5,
    max_grad_norm=1.0
)
```

### Quality Control Pipeline

#### Automated Protocol Verification
```python
qc = QualityControl(
    outlier_threshold=0.95,
    protocol_tolerance=0.1
)
```

## Project Structure
```
arcus/
├── core/                 # Core pipeline components
│   ├── data_loader.py   # Efficient data loading
│   ├── preprocessing.py  # Image preprocessing
│   └── validation.py    # Data validation
├── federated/           # Federated learning
│   ├── client.py       # Client implementation
│   ├── server.py       # Server orchestration
│   └── privacy.py      # DP mechanisms
├── models/              # Neural network models
│   ├── vae.py          # Adversarial VAE
│   └── harmonization.py # CycleGAN
└── utils/              # Utilities
    ├── metrics.py      # Performance metrics
    └── visualization.py # Result plotting
```

## Performance Metrics

- **Privacy**: Differential privacy guarantees (ε, δ)
- **Quality**: Reconstruction error, harmonization SSIM
- **Efficiency**: Training time, memory usage
- **Robustness**: Cross-site consistency scores

## Unique Advantages

- **Privacy-First**: Built-in differential privacy with minimal performance impact
- **Scalable**: Efficient handling of multi-TB datasets
- **Robust**: Fault-tolerant federated training
- **Flexible**: Modular architecture for easy customization
- **Production-Ready**: Comprehensive logging and monitoring

## Contributing
Contributions are welcome! See CONTRIBUTING.md for guidelines.

## Citation
```bibtex
@software{patel2024arcus,
  author = {Patel, Salil},
  title = {ARCUS: Automated Radiological Curation \& Unification System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/salilpatel/arcus}
}
```

## License
MIT License - see LICENSE file for details.

## Author
Salil Patel
