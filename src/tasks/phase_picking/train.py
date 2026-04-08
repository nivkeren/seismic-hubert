"""
Training script for Phase Picking downstream task.

Usage (from project root or src/):
    python src/train_phase_picking.py                                # From project root
    python train_phase_picking.py                                    # From src/
    python train_phase_picking.py pretrained_weights=path/to/model.ckpt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path so imports work from any directory
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader, random_split

from data.stead_dataset import STEADDataset, STEADCollator
from models.seismic_hubert import SeismicHubertConfig
from tasks.phase_picking.model import PhasePickingLightning


class PhasePickingDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for STEAD dataset (Phase Picking)."""
    
    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        channel: str = "all",
        batch_size: int = 16,
        num_workers: int = 4,
        max_samples: int | None = None,
        train_val_split: float = 0.95,
        norm_mode: str = "zscore",
        highpass_freq: float | None = 1.0,
        lowpass_freq: float | None = 40.0,
        label_sigma: float = 10.0,
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
        self.label_sigma = label_sigma
        
        self.train_dataset = None
        self.val_dataset = None
        self.collator = STEADCollator(return_labels=False)
    
    def setup(self, stage: str | None = None):
        full_dataset = STEADDataset(
            hdf5_path=self.hdf5_path,
            csv_path=self.csv_path,
            channel=self.channel,
            max_samples=self.max_samples,
            norm_mode=self.norm_mode,
            highpass_freq=self.highpass_freq,
            lowpass_freq=self.lowpass_freq,
            return_phase_labels=True,
            label_sigma=self.label_sigma,
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function for Phase Picking."""
    
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
    print("\nInitializing data module for phase picking...")
    # For phase picking, we usually need all channels (E, N, Z)
    data_module = PhasePickingDataModule(
        hdf5_path=hdf5_path,
        csv_path=csv_path,
        channel=cfg.data.get("channel", "all"), 
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_samples=cfg.data.max_samples,
        train_val_split=cfg.data.train_val_split,
        norm_mode=cfg.data.norm_mode,
        highpass_freq=cfg.data.highpass_freq,
        lowpass_freq=cfg.data.lowpass_freq,
        label_sigma=10.0, # 10 samples (0.1s at 100Hz) std deviation for label smearing
    )
    data_module.setup()
    
    # ===== Model =====
    print("\nInitializing Phase Picking model...")
    num_channels = 1 if cfg.data.get("channel", "all") == "Z" else 3
    
    model_config = SeismicHubertConfig(
        num_channels=num_channels,
        hidden_size=cfg.model.hidden_size,
        num_hidden_layers=cfg.model.num_layers,
        num_attention_heads=cfg.model.num_heads,
        # Masking is usually turned off for fine-tuning
        mask_prob=0.0,
    )
    
    steps_per_epoch = len(data_module.train_dataset) // (
        cfg.training.batch_size * cfg.training.accumulate_grad_batches
    )
    max_steps = steps_per_epoch * cfg.training.max_epochs
    
    model = PhasePickingLightning(
        config=model_config,
        num_classes=3,  # Noise, P-wave, S-wave
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=max_steps,
        # Determine whether to freeze layers
        freeze_feature_encoder=cfg.get("freeze_feature_encoder", False),
        freeze_base_model=cfg.get("freeze_base_model", False),
        eval_metric=cfg.get("eval_metric", "eqt"),
        tolerance_samples=cfg.get("tolerance_samples", 10),
    )
    
    # Optionally load pretrained base weights
    pretrained_weights = cfg.get("pretrained_weights", None)
    if pretrained_weights:
        weights_path = resolve_path(pretrained_weights)
        print(f"Loading pretrained HuBERT weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        
        # Try to extract the state dict (works with PL checkpoints)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Modify keys if necessary: e.g. mapping "model.hubert.x" to "model.hubert.x"
        base_state_dict = {}
        for key, value in state_dict.items():
            # If loading from the pretraining model, the base model is at `model.hubert`
            if key.startswith("model.hubert."):
                base_state_dict[key] = value
                
        if len(base_state_dict) > 0:
            missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
            print(f"Loaded {len(base_state_dict)} tensors.")
            # Note: We expect the downstream phase-picking head layers to be 'missing'
        else:
            print("Warning: Could not find 'model.hubert' keys in checkpoint.")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== Callbacks =====
    run_name = cfg.logging.run_name or "seismic-hubert-picking"
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
        import mlflow
        mlflow.enable_system_metrics_logging()
        
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
        if callbacks[0].best_model_path:
            mlflow.log_artifact(callbacks[0].best_model_path, artifact_path="checkpoints")
    
    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
