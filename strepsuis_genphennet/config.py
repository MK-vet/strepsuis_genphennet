"""
Configuration module

Handles all configuration parameters for analysis.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for analysis."""

    # Directories
    data_dir: str = "."
    output_dir: str = "./output"

    # Statistical parameters
    bootstrap_iterations: int = 500
    fdr_alpha: float = 0.05
    random_seed: int = 42

    # Reporting parameters
    generate_html: bool = True
    generate_excel: bool = True
    save_png_charts: bool = True
    dpi: int = 150

    # Parallel processing
    n_jobs: int = -1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

        if not 0 < self.fdr_alpha < 1:
            raise ValueError("fdr_alpha must be between 0 and 1")

        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations should be at least 100")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
