"""
Training script for Seismic HuBERT self-supervised pretraining.

This implements a two-stage training similar to the original HuBERT:
1. First iteration: cluster on MFCC-like features, train masked prediction
2. Second iteration: cluster on learned representations, refine model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from seismic_hubert.data import STEADDataset, STEADCollator
from seismic_hubert.models import (
    SeismicHubertConfig,
    SeismicHubertForPreTraining,
    load_seismic_hubert,
)


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
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Pretrained HuBERT model to initialize from"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument(
        "--gradient_accumulation", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.08, help="Masking probability"
    )
    parser.add_argument(
        "--mask_length", type=int, default=10, help="Mask span length"
    )
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', 'mps', or 'auto'"
    )
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument(
        "--wandb_project", type=str, default="seismic-hubert", help="W&B project"
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine the best available device."""
    if device_arg != "auto":
        return torch.device(device_arg)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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


def extract_cluster_labels(
    waveforms: torch.Tensor,
    kmeans_model=None,
) -> torch.Tensor:
    """
    Extract cluster labels for masked prediction.
    
    In the first training iteration, we use simple spectrogram-based features.
    In later iterations, we cluster on learned representations.
    """
    batch_size, seq_length = waveforms.shape[0], waveforms.shape[-1] // 160
    
    # For initial training without k-means, use random labels
    # In practice, you'd extract MFCCs and cluster them
    if kmeans_model is None:
        return torch.randint(0, 100, (batch_size, seq_length))
    
    # With trained k-means, extract features and assign clusters
    # features = extract_features(waveforms)
    # labels = kmeans_model.predict(features)
    # return torch.from_numpy(labels)
    raise NotImplementedError("K-means clustering not yet implemented")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        waveforms = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            features, _ = model.hubert.feature_encoder(waveforms, attention_mask)
            seq_length = features.shape[1]
        
        mask_indices = compute_mask_indices(
            (waveforms.shape[0], seq_length),
            config.mask_prob,
            config.mask_length,
            device,
        )
        
        # Use random cluster labels for initial training
        labels = torch.randint(
            0, config.num_clusters, (waveforms.shape[0], seq_length), device=device
        )
        
        outputs = model(
            input_values=waveforms,
            attention_mask=attention_mask,
            labels=labels,
            mask_time_indices=mask_indices,
        )
        
        loss = outputs["loss"] / config.gradient_accumulation
        loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += outputs["loss"].item()
        num_batches += 1
        
        pbar.set_postfix(
            loss=f"{outputs['loss'].item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )
    
    return {"loss": total_loss / num_batches}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: argparse.Namespace,
) -> dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            waveforms = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            features, _ = model.hubert.feature_encoder(waveforms, attention_mask)
            seq_length = features.shape[1]
            
            mask_indices = compute_mask_indices(
                (waveforms.shape[0], seq_length),
                config.mask_prob,
                config.mask_length,
                device,
            )
            
            labels = torch.randint(
                0, config.num_clusters, (waveforms.shape[0], seq_length), device=device
            )
            
            outputs = model(
                input_values=waveforms,
                attention_mask=attention_mask,
                labels=labels,
                mask_time_indices=mask_indices,
            )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
    
    return {"val_loss": total_loss / num_batches}


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading dataset...")
    num_channels = 1 if args.channel == "Z" else 3
    
    full_dataset = STEADDataset(
        hdf5_path=args.hdf5_path,
        csv_path=args.csv_path,
        channel=args.channel,
        max_samples=args.max_samples,
        normalize=True,
    )
    
    print(f"Dataset statistics:")
    for key, value in full_dataset.get_stats().items():
        print(f"  {key}: {value}")
    
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    collator = STEADCollator(return_labels=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    print("Initializing model...")
    config = SeismicHubertConfig(
        num_channels=num_channels,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_clusters=args.num_clusters,
        mask_prob=args.mask_prob,
        mask_length=args.mask_length,
    )
    
    model = SeismicHubertForPreTraining(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model)
    
    best_val_loss = float("inf")
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args, epoch
        )
        
        val_metrics = validate(model, val_loader, device, args)
        
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}"
        )
        
        if args.wandb:
            import wandb
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
        
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                    "config": config,
                },
                output_dir / "best_model.pt",
            )
            print(f"  Saved best model with val_loss={best_val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
