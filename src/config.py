"""
Configuration management for Seismic HuBERT training.

Supports loading configuration from YAML files with CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Any
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_clusters: int = 100
    

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    hdf5_path: str = ""
    csv_path: str = ""
    channel: Literal["Z", "all"] = "Z"
    max_samples: int | None = None
    norm_mode: str = "zscore"
    highpass_freq: float | None = 1.0
    lowpass_freq: float | None = 40.0
    train_val_split: float = 0.95


@dataclass
class ClusteringConfig:
    """K-means clustering configuration for training targets."""
    
    feature_mode: Literal[
        "spectrogram", "stalta", "frequency_bands", "multi_channel", "combined"
    ] = "spectrogram"
    include_stalta: bool = False
    include_frequency_bands: bool = False
    include_multichannel: bool = False
    feature_dim: int = 32
    hop_length: int = 32


@dataclass
class MaskingConfig:
    """Masking configuration for self-supervised learning."""
    
    mask_prob: float = 0.08
    mask_length: int = 5
    
    # Epoch-based scheduling
    schedule: Literal["constant", "linear", "step", "cosine"] = "constant"
    mask_length_start: int = 3
    mask_length_end: int = 12
    
    # Distance-adaptive masking
    distance_adaptive: bool = False
    distance_mask_min: int = 2
    distance_mask_max: int = 15


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    accumulate_grad_batches: int = 4
    gradient_clip_val: float = 1.0
    
    # System
    num_workers: int = 4
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    seed: int = 42


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
    
    output_dir: str = "outputs"
    
    # MLflow
    mlflow: bool = False
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "seismic-hubert"
    
    # Weights & Biases
    wandb: bool = False
    wandb_project: str = "seismic-hubert"


@dataclass
class TrainConfig:
    """Complete training configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Resume from checkpoint
    resume_from: str | None = None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data or {})
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Create configuration from a dictionary."""
        config = cls()
        
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "clustering" in data:
            config.clustering = ClusteringConfig(**data["clustering"])
        if "masking" in data:
            config.masking = MaskingConfig(**data["masking"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "resume_from" in data:
            config.resume_from = data["resume_from"]
        
        return config
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "clustering": asdict(self.clustering),
            "masking": asdict(self.masking),
            "training": asdict(self.training),
            "logging": asdict(self.logging),
            "resume_from": self.resume_from,
        }
    
    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def merge_cli_args(self, args) -> "TrainConfig":
        """Merge CLI arguments into config (CLI takes precedence)."""
        # Data paths (required, always override)
        if hasattr(args, 'hdf5_path') and args.hdf5_path:
            self.data.hdf5_path = args.hdf5_path
        if hasattr(args, 'csv_path') and args.csv_path:
            self.data.csv_path = args.csv_path
        
        # Only override if explicitly set (not default)
        # This is tricky with argparse, so we override all CLI args
        if hasattr(args, 'channel'):
            self.data.channel = args.channel
        if hasattr(args, 'max_samples') and args.max_samples is not None:
            self.data.max_samples = args.max_samples
        
        # Model
        if hasattr(args, 'hidden_size'):
            self.model.hidden_size = args.hidden_size
        if hasattr(args, 'num_layers'):
            self.model.num_layers = args.num_layers
        if hasattr(args, 'num_heads'):
            self.model.num_heads = args.num_heads
        if hasattr(args, 'num_clusters'):
            self.model.num_clusters = args.num_clusters
        
        # Clustering
        if hasattr(args, 'feature_mode'):
            self.clustering.feature_mode = args.feature_mode
        if hasattr(args, 'include_stalta'):
            self.clustering.include_stalta = args.include_stalta
        if hasattr(args, 'include_frequency_bands'):
            self.clustering.include_frequency_bands = args.include_frequency_bands
        if hasattr(args, 'include_multichannel'):
            self.clustering.include_multichannel = args.include_multichannel
        
        # Masking
        if hasattr(args, 'mask_prob'):
            self.masking.mask_prob = args.mask_prob
        if hasattr(args, 'mask_length'):
            self.masking.mask_length = args.mask_length
        if hasattr(args, 'mask_schedule'):
            self.masking.schedule = args.mask_schedule
        if hasattr(args, 'mask_length_start'):
            self.masking.mask_length_start = args.mask_length_start
        if hasattr(args, 'mask_length_end'):
            self.masking.mask_length_end = args.mask_length_end
        if hasattr(args, 'distance_adaptive_mask'):
            self.masking.distance_adaptive = args.distance_adaptive_mask
        if hasattr(args, 'distance_mask_min'):
            self.masking.distance_mask_min = args.distance_mask_min
        if hasattr(args, 'distance_mask_max'):
            self.masking.distance_mask_max = args.distance_mask_max
        
        # Training
        if hasattr(args, 'batch_size'):
            self.training.batch_size = args.batch_size
        if hasattr(args, 'lr'):
            self.training.lr = args.lr
        if hasattr(args, 'weight_decay'):
            self.training.weight_decay = args.weight_decay
        if hasattr(args, 'max_epochs'):
            self.training.max_epochs = args.max_epochs
        if hasattr(args, 'warmup_steps'):
            self.training.warmup_steps = args.warmup_steps
        if hasattr(args, 'accumulate_grad_batches'):
            self.training.accumulate_grad_batches = args.accumulate_grad_batches
        if hasattr(args, 'gradient_clip_val'):
            self.training.gradient_clip_val = args.gradient_clip_val
        if hasattr(args, 'num_workers'):
            self.training.num_workers = args.num_workers
        if hasattr(args, 'accelerator'):
            self.training.accelerator = args.accelerator
        if hasattr(args, 'devices'):
            self.training.devices = args.devices
        if hasattr(args, 'precision'):
            self.training.precision = args.precision
        if hasattr(args, 'seed'):
            self.training.seed = args.seed
        
        # Logging
        if hasattr(args, 'output_dir'):
            self.logging.output_dir = args.output_dir
        if hasattr(args, 'mlflow'):
            self.logging.mlflow = args.mlflow
        if hasattr(args, 'mlflow_tracking_uri'):
            self.logging.mlflow_tracking_uri = args.mlflow_tracking_uri
        if hasattr(args, 'mlflow_experiment'):
            self.logging.mlflow_experiment = args.mlflow_experiment
        if hasattr(args, 'wandb'):
            self.logging.wandb = args.wandb
        if hasattr(args, 'wandb_project'):
            self.logging.wandb_project = args.wandb_project
        
        # Resume
        if hasattr(args, 'resume_from') and args.resume_from:
            self.resume_from = args.resume_from
        
        return self


def generate_default_config(path: str | Path = "config/default.yaml") -> None:
    """Generate a default configuration file."""
    config = TrainConfig()
    config.save_yaml(path)
    print(f"Generated default config at: {path}")
