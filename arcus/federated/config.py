"""Federated learning configuration."""

from dataclasses import dataclass


@dataclass
class FederatedConfig:
    """Configuration for federated learning.
    
    Attributes:
        min_site_samples: Minimum number of samples required per site
        rounds: Number of federated training rounds
        local_epochs: Number of local training epochs per round
        min_sites: Minimum number of sites required for federation
        aggregation_method: Method for aggregating model updates
    """
    min_site_samples: int = 100
    rounds: int = 10
    local_epochs: int = 5
    min_sites: int = 2
    aggregation_method: str = "fedavg"  # or "fedprox"
