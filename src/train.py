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
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from data import STEADDataset, STEADCollator, ClusterLabelGenerator, align_labels_to_features
from models.seismic_hubert import (
    SeismicHubertConfig,
    SeismicHubertForPreTraining,
)


def compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int,
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
    mask_length : int
        Length of each mask span
    device : torch.device
        Device to create tensor on
    
    Returns
    -------
    torch.Tensor
        Boolean mask of shape (batch_size, sequence_length)
    """
    batch_size, seq_length = shape
    
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    
    num_masked_spans = max(1, int(mask_prob * seq_length / mask_length))
    
    for batch_idx in range(batch_size):
        span_starts = torch.randint(
            0, max(1, seq_length - mask_length), (num_masked_spans,)
        )
        for start in span_starts:
            mask[batch_idx, start : start + mask_length] = True
    
    return mask


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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config", "label_generator"])
        
        self.config = config
        self.label_generator = label_generator
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        self.model = SeismicHubertForPreTraining(config)
    
    def forward(self, input_values, attention_mask=None, labels=None, mask_time_indices=None):
        return self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            mask_time_indices=mask_time_indices,
        )
    
    def _shared_step(self, batch, batch_idx):
        """Shared logic for training and validation steps."""
        waveforms = batch["input_values"]
        attention_mask = batch["attention_mask"]
        
        # Get sequence length from feature encoder
        with torch.no_grad():
            features, _ = self.model.hubert.feature_encoder(waveforms, attention_mask)
            seq_length = features.shape[1]
        
        # Compute mask indices
        mask_indices = compute_mask_indices(
            (waveforms.shape[0], seq_length),
            self.config.mask_prob,
            self.config.mask_length,
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
    
    # Data arguments
    parser.add_argument(
        "--hdf5_path", type=str, required=True, help="Path to STEAD HDF5 file"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to STEAD CSV file"
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
        "--mask_length", type=int, default=10, help="Mask span length"
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
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data module
    print("Initializing data module...")
    data_module = STEADDataModule(
        hdf5_path=args.hdf5_path,
        csv_path=args.csv_path,
        channel=args.channel,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=STEADCollator(),
        )
        
        label_generator = ClusterLabelGenerator(
            n_clusters=args.num_clusters,
            feature_dim=32,
            hop_length=32,
            sample_rate=100,
        )
        label_generator.fit(kmeans_loader, max_samples=min(10000, len(data_module.train_dataset)))
        label_generator.save(kmeans_path)
        print(f"K-means model saved to {kmeans_path}")
    
    # Initialize model
    print("\nInitializing model...")
    num_channels = 1 if args.channel == "Z" else 3
    
    config = SeismicHubertConfig(
        num_channels=num_channels,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_clusters=args.num_clusters,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
    )
    
    # Estimate max steps for scheduler
    steps_per_epoch = len(data_module.train_dataset) // (args.batch_size * args.accumulate_grad_batches)
    max_steps = steps_per_epoch * args.max_epochs
    
    model = SeismicHubertLightning(
        config=config,
        label_generator=label_generator,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
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
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
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
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
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
        ckpt_path=args.resume_from,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model checkpoint: {callbacks[0].best_model_path}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
