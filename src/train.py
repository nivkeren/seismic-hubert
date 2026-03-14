"""
Training script for Seismic HuBERT self-supervised pretraining using PyTorch Lightning.

This implements a two-stage training similar to the original HuBERT:
1. First iteration: cluster on MFCC-like features, train masked prediction
2. Second iteration: cluster on learned representations, refine model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader, random_split

from data import STEADDataset, STEADCollator, ClusterLabelGenerator, align_labels_to_features
from models.seismic_hubert import (
    SeismicHubertConfig,
    SeismicHubertForPreTraining,
)


def compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int | list[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute random mask spans for self-supervised learning.
    
    Parameters
    ----------
    shape : tuple[int, int]
        (batch_size, sequence_length)
    mask_prob : float
        Probability of masking each position
    mask_length : int or list[int] or torch.Tensor
        Length of each mask span. Can be a single value for all samples,
        or per-sample values for distance-adaptive masking.
    device : torch.device
        Device to create tensor on
    
    Returns
    -------
    torch.Tensor
        Boolean mask of shape (batch_size, sequence_length)
    """
    batch_size, seq_length = shape
    
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    
    # Handle per-sample mask lengths
    if isinstance(mask_length, (list, torch.Tensor)):
        mask_lengths = mask_length if isinstance(mask_length, list) else mask_length.tolist()
    else:
        mask_lengths = [mask_length] * batch_size
    
    for batch_idx in range(batch_size):
        ml = int(mask_lengths[batch_idx])
        num_masked_spans = max(1, int(mask_prob * seq_length / ml))
        
        span_starts = torch.randint(
            0, max(1, seq_length - ml), (num_masked_spans,)
        )
        for start in span_starts:
            mask[batch_idx, start : start + ml] = True
    
    return mask


def compute_distance_adaptive_mask_length(
    distances_km: torch.Tensor,
    min_mask: int = 2,
    max_mask: int = 15,
    min_distance: float = 10.0,
    max_distance: float = 200.0,
) -> torch.Tensor:
    """
    Compute mask length based on source distance.
    
    Closer events have shorter P-to-S intervals, so they need shorter masks.
    P-to-S time ≈ distance / 8 km/s (rough approximation).
    
    Parameters
    ----------
    distances_km : torch.Tensor
        Source distances in km. Negative values indicate noise (no distance).
    min_mask : int
        Minimum mask length (for closest events)
    max_mask : int
        Maximum mask length (for distant events)
    min_distance : float
        Distance below which min_mask is used
    max_distance : float
        Distance above which max_mask is used
    
    Returns
    -------
    torch.Tensor
        Per-sample mask lengths
    """
    # Clamp distances to valid range
    distances = distances_km.clone()
    
    # For noise samples (distance < 0), use middle mask length
    noise_mask = distances < 0
    distances = distances.clamp(min_distance, max_distance)
    
    # Linear interpolation based on distance
    progress = (distances - min_distance) / (max_distance - min_distance)
    mask_lengths = min_mask + progress * (max_mask - min_mask)
    
    # Noise samples get average mask length
    avg_mask = (min_mask + max_mask) / 2
    mask_lengths[noise_mask] = avg_mask
    
    return mask_lengths.round().int()


class STEADDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for STEAD dataset."""
    
    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        channel: str = "Z",
        batch_size: int = 16,
        num_workers: int = 4,
        max_samples: int | None = None,
        train_val_split: float = 0.95,
        norm_mode: str = "zscore",
        highpass_freq: float | None = 1.0,
        lowpass_freq: float | None = 40.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.hdf5_path = hdf5_path
        self.csv_path = csv_path
        self.channel = channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.train_val_split = train_val_split
        self.norm_mode = norm_mode
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        
        self.train_dataset = None
        self.val_dataset = None
        self.collator = STEADCollator(return_labels=True)
    
    def setup(self, stage: str | None = None):
        """Set up datasets for training and validation."""
        full_dataset = STEADDataset(
            hdf5_path=self.hdf5_path,
            csv_path=self.csv_path,
            channel=self.channel,
            max_samples=self.max_samples,
            norm_mode=self.norm_mode,
            highpass_freq=self.highpass_freq,
            lowpass_freq=self.lowpass_freq,
        )
        
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Dataset: {len(full_dataset)} total, {train_size} train, {val_size} val")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class SeismicHubertLightning(pl.LightningModule):
    """PyTorch Lightning module for Seismic HuBERT pretraining."""
    
    def __init__(
        self,
        config: SeismicHubertConfig,
        label_generator: ClusterLabelGenerator,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        max_epochs: int = 100,
        mask_length_start: int | None = None,
        mask_length_end: int | None = None,
        mask_schedule: str = "constant",
        distance_adaptive_mask: bool = False,
        distance_mask_min: int = 2,
        distance_mask_max: int = 15,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "label_generator"])
        
        self.config = config
        self.label_generator = label_generator
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        
        # Mask scheduling (epoch-based)
        self.mask_schedule = mask_schedule
        self.mask_length_start = mask_length_start or config.mask_length
        self.mask_length_end = mask_length_end or config.mask_length
        self._current_mask_length = self.mask_length_start
        
        # Distance-adaptive masking (per-sample)
        self.distance_adaptive_mask = distance_adaptive_mask
        self.distance_mask_min = distance_mask_min
        self.distance_mask_max = distance_mask_max
        
        self.model = SeismicHubertForPreTraining(config)
    
    def forward(self, input_values, attention_mask=None, labels=None, mask_time_indices=None):
        return self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            mask_time_indices=mask_time_indices,
        )
    
    def get_current_mask_length(self) -> int:
        """Compute mask length based on schedule and current epoch."""
        if self.mask_schedule == "constant":
            return self.mask_length_start
        
        # Get progress through training (0.0 to 1.0)
        current_epoch = self.current_epoch
        progress = min(1.0, current_epoch / max(1, self.max_epochs - 1))
        
        if self.mask_schedule == "linear":
            # Linear interpolation from start to end
            mask_length = self.mask_length_start + progress * (
                self.mask_length_end - self.mask_length_start
            )
        elif self.mask_schedule == "step":
            # Step schedule: 3 stages at 0%, 33%, 66%
            if progress < 0.33:
                mask_length = self.mask_length_start
            elif progress < 0.66:
                mask_length = (self.mask_length_start + self.mask_length_end) / 2
            else:
                mask_length = self.mask_length_end
        elif self.mask_schedule == "cosine":
            # Cosine schedule: slower at start and end
            cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
            mask_length = self.mask_length_start + cosine_progress * (
                self.mask_length_end - self.mask_length_start
            )
        else:
            mask_length = self.mask_length_start
        
        return max(1, int(round(mask_length)))
    
    def on_train_epoch_start(self):
        """Update mask length at the start of each epoch."""
        self._current_mask_length = self.get_current_mask_length()
        self.log("mask_length", float(self._current_mask_length), prog_bar=True)
    
    def _shared_step(self, batch, batch_idx):
        """Shared logic for training and validation steps."""
        waveforms = batch["input_values"]
        attention_mask = batch["attention_mask"]
        
        # Get sequence length from feature encoder
        with torch.no_grad():
            features, _ = self.model.hubert.feature_encoder(waveforms, attention_mask)
            seq_length = features.shape[1]
        
        # Determine mask length(s)
        if self.distance_adaptive_mask and "source_distance_km" in batch:
            # Per-sample mask lengths based on distance
            distances = batch["source_distance_km"]
            
            # Scale distance-based masks by epoch progress (curriculum)
            if self.mask_schedule != "constant":
                epoch_progress = min(1.0, self.current_epoch / max(1, self.max_epochs - 1))
                # Scale the max mask length based on epoch progress
                current_max = self.distance_mask_min + epoch_progress * (
                    self.distance_mask_max - self.distance_mask_min
                )
            else:
                current_max = self.distance_mask_max
            
            mask_lengths = compute_distance_adaptive_mask_length(
                distances,
                min_mask=self.distance_mask_min,
                max_mask=int(current_max),
            )
        else:
            # Single mask length for all samples
            mask_lengths = self._current_mask_length
        
        # Compute mask indices
        mask_indices = compute_mask_indices(
            (waveforms.shape[0], seq_length),
            self.config.mask_prob,
            mask_lengths,
            waveforms.device,
        )
        
        # Get cluster labels from spectrogram features
        labels = self.label_generator.get_labels(batch["input_values"].cpu())
        labels = align_labels_to_features(labels, seq_length)
        labels = labels.to(waveforms.device)
        
        # Forward pass
        outputs = self.model(
            input_values=waveforms,
            attention_mask=attention_mask,
            labels=labels,
            mask_time_indices=mask_indices,
        )
        
        return outputs["loss"]
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Cosine annealing with warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Seismic HuBERT")
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override config file values."
    )
    
    # Data arguments
    parser.add_argument(
        "--hdf5_path", type=str, default=None, help="Path to STEAD HDF5 file"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None, help="Path to STEAD CSV file"
    )
    parser.add_argument(
        "--channel", type=str, default="Z", choices=["Z", "all"],
        help="Seismic channels to use"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples (for debugging)"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Transformer hidden size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=12, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=100,
        help="Number of K-means clusters for targets"
    )
    
    # Clustering feature arguments (domain-specific)
    parser.add_argument(
        "--feature_mode", type=str, default="spectrogram",
        choices=["spectrogram", "stalta", "frequency_bands", "multi_channel", "combined"],
        help="Feature extraction mode for K-means clustering"
    )
    parser.add_argument(
        "--include_stalta", action="store_true",
        help="Include STA/LTA features (when feature_mode=combined)"
    )
    parser.add_argument(
        "--include_frequency_bands", action="store_true",
        help="Include frequency band features (when feature_mode=combined)"
    )
    parser.add_argument(
        "--include_multichannel", action="store_true",
        help="Include polarization features (when feature_mode=combined, requires channel=all)"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.08, help="Masking probability"
    )
    parser.add_argument(
        "--mask_length", type=int, default=5,
        help="Mask span length in frames (~0.32s each). Used when mask_schedule=constant."
    )
    parser.add_argument(
        "--mask_length_start", type=int, default=3,
        help="Starting mask length for scheduled masking"
    )
    parser.add_argument(
        "--mask_length_end", type=int, default=12,
        help="Ending mask length for scheduled masking"
    )
    parser.add_argument(
        "--mask_schedule", type=str, default="constant",
        choices=["constant", "linear", "step", "cosine"],
        help="Mask length schedule: constant, linear, step (3 stages), or cosine"
    )
    parser.add_argument(
        "--distance_adaptive_mask", action="store_true",
        help="Enable per-sample mask length based on source distance"
    )
    parser.add_argument(
        "--distance_mask_min", type=int, default=2,
        help="Minimum mask length for close events (distance-adaptive)"
    )
    parser.add_argument(
        "--distance_mask_max", type=int, default=15,
        help="Maximum mask length for distant events (distance-adaptive)"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="Accelerator: 'cpu', 'gpu', 'mps', or 'auto'"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices"
    )
    parser.add_argument(
        "--precision", type=str, default="32",
        help="Training precision: '32', '16-mixed', 'bf16-mixed'"
    )
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument(
        "--wandb_project", type=str, default="seismic-hubert", help="W&B project"
    )
    parser.add_argument("--mlflow", action="store_true", help="Use MLflow tracking")
    parser.add_argument(
        "--mlflow_tracking_uri", type=str, default="mlruns",
        help="MLflow tracking URI (local path or server URL)"
    )
    parser.add_argument(
        "--mlflow_experiment", type=str, default="seismic-hubert",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        from config import TrainConfig
        print(f"Loading config from: {args.config}")
        cfg = TrainConfig.from_yaml(args.config)
        cfg.merge_cli_args(args)
    else:
        # Create config from CLI args only
        from config import TrainConfig
        cfg = TrainConfig()
        cfg.merge_cli_args(args)
    
    # Validate required paths
    if not cfg.data.hdf5_path or not cfg.data.csv_path:
        raise ValueError("hdf5_path and csv_path are required (via --config or CLI)")
    
    # Set seed
    pl.seed_everything(cfg.training.seed)
    
    # Create output directory
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save effective config
    cfg.save_yaml(output_dir / "config.yaml")
    print(f"Config saved to: {output_dir / 'config.yaml'}")
    
    # Initialize data module
    print("\nInitializing data module...")
    data_module = STEADDataModule(
        hdf5_path=cfg.data.hdf5_path,
        csv_path=cfg.data.csv_path,
        channel=cfg.data.channel,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_samples=cfg.data.max_samples,
    )
    data_module.setup()
    
    # Initialize K-means clustering for targets
    print("\nInitializing K-means clustering for training targets...")
    kmeans_path = output_dir / "kmeans.pkl"
    
    if kmeans_path.exists():
        print(f"Loading existing K-means model from {kmeans_path}")
        label_generator = ClusterLabelGenerator.load(kmeans_path)
    else:
        kmeans_loader = DataLoader(
            data_module.train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=STEADCollator(),
        )
        
        label_generator = ClusterLabelGenerator(
            n_clusters=cfg.model.num_clusters,
            feature_dim=cfg.clustering.feature_dim,
            hop_length=cfg.clustering.hop_length,
            sample_rate=100,
            feature_mode=cfg.clustering.feature_mode,
            include_stalta=cfg.clustering.include_stalta,
            include_frequency_bands=cfg.clustering.include_frequency_bands,
            include_multichannel=cfg.clustering.include_multichannel,
        )
        print(f"Using feature mode: {cfg.clustering.feature_mode}")
        label_generator.fit(kmeans_loader, max_samples=min(10000, len(data_module.train_dataset)))
        label_generator.save(kmeans_path)
        print(f"K-means model saved to {kmeans_path}")
    
    # Initialize model
    print("\nInitializing model...")
    num_channels = 1 if cfg.data.channel == "Z" else 3
    
    model_config = SeismicHubertConfig(
        num_channels=num_channels,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_layers,
        num_attention_heads=cfg.model.num_heads,
        num_clusters=cfg.model.num_clusters,
        mask_prob=cfg.masking.mask_prob,
        mask_length=cfg.masking.mask_length,
    )
    
    # Estimate max steps for scheduler
    steps_per_epoch = len(data_module.train_dataset) // (
        cfg.training.batch_size * cfg.training.accumulate_grad_batches
    )
    max_steps = steps_per_epoch * cfg.training.max_epochs
    
    # Determine mask length parameters
    if cfg.masking.schedule == "constant" and not cfg.masking.distance_adaptive:
        mask_start = cfg.masking.mask_length
        mask_end = cfg.masking.mask_length
    else:
        mask_start = cfg.masking.mask_length_start
        mask_end = cfg.masking.mask_length_end
        if cfg.masking.schedule != "constant":
            print(f"Mask scheduling: {cfg.masking.schedule} from {mask_start} to {mask_end} frames")
    
    if cfg.masking.distance_adaptive:
        print(f"Distance-adaptive masking: {cfg.masking.distance_mask_min}-{cfg.masking.distance_mask_max} frames")
    
    model = SeismicHubertLightning(
        config=model_config,
        label_generator=label_generator,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=max_steps,
        max_epochs=cfg.training.max_epochs,
        mask_length_start=mask_start,
        mask_length_end=mask_end,
        mask_schedule=cfg.masking.schedule,
        distance_adaptive_mask=cfg.masking.distance_adaptive,
        distance_mask_min=cfg.masking.distance_mask_min,
        distance_mask_max=cfg.masking.distance_mask_max,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="seismic-hubert-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
    ]
    
    # Logger
    if cfg.logging.mlflow:
        logger = MLFlowLogger(
            experiment_name=cfg.logging.mlflow_experiment,
            tracking_uri=cfg.logging.mlflow_tracking_uri,
            log_model=True,
            tags={
                "model": "seismic-hubert",
                "channel": cfg.data.channel,
                "hidden_size": str(cfg.model.hidden_size),
                "num_layers": str(cfg.model.num_layers),
            },
        )
        # Log hyperparameters
        logger.log_hyperparams(cfg.to_dict())
    elif cfg.logging.wandb:
        logger = WandbLogger(
            project=cfg.logging.wandb_project,
            save_dir=output_dir,
            log_model=True,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard",
        )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=output_dir,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=cfg.resume_from,
    )
    
    # Log artifacts to MLflow
    if cfg.logging.mlflow:
        import mlflow
        
        # Log K-means model
        if kmeans_path.exists():
            mlflow.log_artifact(str(kmeans_path), artifact_path="kmeans")
        
        # Log best checkpoint
        best_ckpt = callbacks[0].best_model_path
        if best_ckpt:
            mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
        
        # Log config
        mlflow.log_artifact(str(output_dir / "config.yaml"), artifact_path="config")
    
    print(f"\nTraining complete!")
    print(f"Best model checkpoint: {callbacks[0].best_model_path}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
