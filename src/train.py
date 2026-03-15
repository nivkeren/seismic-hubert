"""
Training script for Seismic HuBERT self-supervised pretraining.

Usage (from project root or src/):
    python src/train.py                              # From project root
    python train.py                                  # From src/
    python train.py +experiment=overfit              # Use experiment preset
    python train.py training.max_epochs=50           # Override any value
    python train.py --multirun training.lr=1e-4,5e-5 # Hyperparameter sweep
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path so imports work from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader, random_split

from data import STEADDataset, STEADCollator, ClusterLabelGenerator, align_labels_to_features
from models.seismic_hubert import SeismicHubertConfig, SeismicHubertForPreTraining


def compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int | list[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute random mask spans for self-supervised learning."""
    batch_size, seq_length = shape
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
    
    if isinstance(mask_length, (list, torch.Tensor)):
        mask_lengths = mask_length if isinstance(mask_length, list) else mask_length.tolist()
    else:
        mask_lengths = [mask_length] * batch_size
    
    for batch_idx in range(batch_size):
        ml = int(mask_lengths[batch_idx])
        num_masked_spans = max(1, int(mask_prob * seq_length / ml))
        span_starts = torch.randint(0, max(1, seq_length - ml), (num_masked_spans,))
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
    """Compute mask length based on source distance."""
    distances = distances_km.clone()
    noise_mask = distances < 0
    distances = distances.clamp(min_distance, max_distance)
    progress = (distances - min_distance) / (max_distance - min_distance)
    mask_lengths = min_mask + progress * (max_mask - min_mask)
    mask_lengths[noise_mask] = (min_mask + max_mask) / 2
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
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
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
        
        self.mask_schedule = mask_schedule
        self.mask_length_start = mask_length_start or config.mask_length
        self.mask_length_end = mask_length_end or config.mask_length
        self._current_mask_length = self.mask_length_start
        
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
        if self.mask_schedule == "constant":
            return self.mask_length_start
        
        progress = min(1.0, self.current_epoch / max(1, self.max_epochs - 1))
        
        if self.mask_schedule == "linear":
            mask_length = self.mask_length_start + progress * (self.mask_length_end - self.mask_length_start)
        elif self.mask_schedule == "step":
            if progress < 0.33:
                mask_length = self.mask_length_start
            elif progress < 0.66:
                mask_length = (self.mask_length_start + self.mask_length_end) / 2
            else:
                mask_length = self.mask_length_end
        elif self.mask_schedule == "cosine":
            cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
            mask_length = self.mask_length_start + cosine_progress * (self.mask_length_end - self.mask_length_start)
        else:
            mask_length = self.mask_length_start
        
        return max(1, int(round(mask_length)))
    
    def on_train_epoch_start(self):
        self._current_mask_length = self.get_current_mask_length()
        self.log("mask_length", float(self._current_mask_length), prog_bar=True)
    
    def _shared_step(self, batch, batch_idx):
        waveforms = batch["input_values"]
        attention_mask = batch["attention_mask"]
        
        with torch.no_grad():
            features, _ = self.model.hubert.feature_encoder(waveforms, attention_mask)
            seq_length = features.shape[1]
        
        if self.distance_adaptive_mask and "source_distance_km" in batch:
            distances = batch["source_distance_km"]
            if self.mask_schedule != "constant":
                epoch_progress = min(1.0, self.current_epoch / max(1, self.max_epochs - 1))
                current_max = self.distance_mask_min + epoch_progress * (self.distance_mask_max - self.distance_mask_min)
            else:
                current_max = self.distance_mask_max
            mask_lengths = compute_distance_adaptive_mask_length(
                distances, min_mask=self.distance_mask_min, max_mask=int(current_max)
            )
        else:
            mask_lengths = self._current_mask_length
        
        mask_indices = compute_mask_indices(
            (waveforms.shape[0], seq_length),
            self.config.mask_prob,
            mask_lengths,
            waveforms.device,
        )
        
        labels = self.label_generator.get_labels(batch["input_values"].cpu())
        labels = align_labels_to_features(labels, seq_length)
        labels = labels.to(waveforms.device)
        
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
        
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    pl.seed_everything(cfg.training.seed)
    
    # Output directory (managed by Hydra)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Resolve data paths relative to project root
    def resolve_path(path_str: str) -> str:
        """Resolve path relative to project root if not absolute."""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str(PROJECT_ROOT / p)
    
    hdf5_path = resolve_path(cfg.data.hdf5_path)
    csv_path = resolve_path(cfg.data.csv_path)
    
    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # ===== Data Module =====
    print("\nInitializing data module...")
    data_module = STEADDataModule(
        hdf5_path=hdf5_path,
        csv_path=csv_path,
        channel=cfg.data.channel,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_samples=cfg.data.max_samples,
        train_val_split=cfg.data.train_val_split,
        norm_mode=cfg.data.norm_mode,
        highpass_freq=cfg.data.highpass_freq,
        lowpass_freq=cfg.data.lowpass_freq,
    )
    data_module.setup()
    
    # ===== K-means Clustering =====
    print("\nInitializing K-means clustering...")
    kmeans_path = output_dir / "kmeans.pkl"
    
    if kmeans_path.exists():
        print(f"Loading existing K-means model from {kmeans_path}")
        label_generator = ClusterLabelGenerator.load(kmeans_path)
    else:
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
        print(f"Feature mode: {cfg.clustering.feature_mode}")
        
        kmeans_loader = DataLoader(
            data_module.train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=STEADCollator(),
        )
        label_generator.fit(kmeans_loader, max_samples=min(10000, len(data_module.train_dataset)))
        label_generator.save(kmeans_path)
        print(f"K-means saved to {kmeans_path}")
    
    # ===== Model =====
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
    
    steps_per_epoch = len(data_module.train_dataset) // (
        cfg.training.batch_size * cfg.training.accumulate_grad_batches
    )
    max_steps = steps_per_epoch * cfg.training.max_epochs
    
    # Mask parameters
    if cfg.masking.schedule == "constant" and not cfg.masking.distance_adaptive:
        mask_start, mask_end = cfg.masking.mask_length, cfg.masking.mask_length
    else:
        mask_start, mask_end = cfg.masking.mask_length_start, cfg.masking.mask_length_end
        if cfg.masking.schedule != "constant":
            print(f"Mask scheduling: {cfg.masking.schedule} ({mask_start} -> {mask_end})")
    
    if cfg.masking.distance_adaptive:
        print(f"Distance-adaptive masking: {cfg.masking.distance_mask_min}-{cfg.masking.distance_mask_max}")
    
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
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== Callbacks =====
    run_name = cfg.logging.run_name or "seismic-hubert"
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename=f"{run_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
    ]
    
    # ===== Logger =====
    if cfg.logging.logger == "mlflow":
        logger = MLFlowLogger(
            experiment_name=cfg.logging.mlflow_experiment,
            tracking_uri=cfg.logging.mlflow_tracking_uri,
            run_name=run_name,
            log_model=True,
        )
    elif cfg.logging.logger == "wandb":
        logger = WandbLogger(
            project=cfg.logging.wandb_project,
            name=run_name,
            save_dir=str(output_dir),
            log_model=True,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(output_dir),
            name="tensorboard",
            version=run_name,
        )
    
    # ===== Trainer =====
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_dir),
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
    )
    
    # ===== Train =====
    print(f"\nStarting training...")
    resume_ckpt = resolve_path(cfg.resume_from) if cfg.resume_from else None
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    
    # Log artifacts for MLflow
    if cfg.logging.logger == "mlflow":
        import mlflow
        if kmeans_path.exists():
            mlflow.log_artifact(str(kmeans_path), artifact_path="kmeans")
        if callbacks[0].best_model_path:
            mlflow.log_artifact(callbacks[0].best_model_path, artifact_path="checkpoints")
    
    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
