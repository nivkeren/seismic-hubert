"""
K-means clustering for HuBERT-style self-supervised learning targets.

This module provides functionality to:
1. Extract spectrogram features from seismic waveforms
2. Run K-means clustering on these features
3. Generate cluster labels for training targets
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from scipy import signal
from tqdm import tqdm
from pathlib import Path
import pickle
from typing import Optional


def extract_spectrogram_features(
    waveform: np.ndarray,
    sample_rate: int = 100,
    n_fft: int = 64,
    hop_length: int = 32,
    n_mels: int = 40,
) -> np.ndarray:
    """
    Extract spectrogram features from a waveform.
    
    Uses a simple STFT-based spectrogram suitable for seismic data.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform of shape (channels, samples) or (samples,)
    sample_rate : int
        Sample rate in Hz
    n_fft : int
        FFT window size
    hop_length : int
        Hop length between frames
    n_mels : int
        Number of mel bands (or frequency bins to keep)
    
    Returns
    -------
    np.ndarray
        Spectrogram features of shape (frames, features)
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    
    # Use first channel if multi-channel
    wav = waveform[0] if waveform.shape[0] <= 3 else waveform
    
    # Compute STFT
    f, t, Zxx = signal.stft(wav, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # Take magnitude and convert to log scale
    magnitude = np.abs(Zxx)
    log_spec = np.log(magnitude + 1e-10)
    
    # Keep only lower frequency bins (most relevant for seismic)
    n_bins = min(n_mels, log_spec.shape[0])
    log_spec = log_spec[:n_bins, :]
    
    # Transpose to (frames, features)
    features = log_spec.T
    
    return features.astype(np.float32)


class ClusterLabelGenerator:
    """
    Generates cluster labels for HuBERT training.
    
    This class handles:
    1. Extracting spectrogram features from waveforms
    2. Training K-means clustering
    3. Assigning cluster labels to new waveforms
    """
    
    def __init__(
        self,
        n_clusters: int = 100,
        feature_dim: int = 32,
        hop_length: int = 32,
        sample_rate: int = 100,
    ):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.kmeans: Optional[MiniBatchKMeans] = None
        self._fitted = False
    
    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract features from a single waveform."""
        return extract_spectrogram_features(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.feature_dim * 2,
            hop_length=self.hop_length,
            n_mels=self.feature_dim,
        )
    
    def fit(
        self,
        dataloader: DataLoader,
        max_samples: int = 50000,
        verbose: bool = True,
    ) -> "ClusterLabelGenerator":
        """
        Fit K-means clustering on features from the dataset.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing waveform samples
        max_samples : int
            Maximum number of samples to use for fitting
        verbose : bool
            Whether to show progress bar
        
        Returns
        -------
        self
        """
        if verbose:
            print(f"Extracting features for K-means clustering (n_clusters={self.n_clusters})...")
        
        all_features = []
        n_samples = 0
        
        iterator = tqdm(dataloader, desc="Extracting features") if verbose else dataloader
        
        for batch in iterator:
            # Handle both collated (input_values) and raw (waveform) batch formats
            if "input_values" in batch:
                waveforms = batch["input_values"].numpy()
            else:
                waveforms = batch["waveform"].numpy()
            
            for waveform in waveforms:
                features = self.extract_features(waveform)
                all_features.append(features)
                n_samples += 1
                
                if n_samples >= max_samples:
                    break
            
            if n_samples >= max_samples:
                break
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        
        if verbose:
            print(f"Fitting K-means on {len(all_features)} feature frames...")
        
        # Fit K-means
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=1024,
            n_init=3,
            random_state=42,
            verbose=verbose,
        )
        self.kmeans.fit(all_features)
        self._fitted = True
        
        if verbose:
            print(f"K-means clustering complete. Inertia: {self.kmeans.inertia_:.2f}")
        
        return self
    
    def get_labels(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Get cluster labels for a waveform.
        
        Parameters
        ----------
        waveform : torch.Tensor or np.ndarray
            Waveform of shape (channels, samples) or (batch, channels, samples)
        
        Returns
        -------
        torch.Tensor
            Cluster labels of shape (frames,) or (batch, frames)
        """
        if not self._fitted:
            raise RuntimeError("ClusterLabelGenerator not fitted. Call fit() first.")
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # Handle batch dimension
        if waveform.ndim == 3:
            # Batch of waveforms
            labels_list = []
            for wav in waveform:
                features = self.extract_features(wav)
                labels = self.kmeans.predict(features)
                labels_list.append(labels)
            
            # Pad to same length if needed
            max_len = max(len(l) for l in labels_list)
            labels_padded = np.zeros((len(labels_list), max_len), dtype=np.int64)
            for i, labels in enumerate(labels_list):
                labels_padded[i, :len(labels)] = labels
            
            return torch.from_numpy(labels_padded)
        else:
            # Single waveform
            features = self.extract_features(waveform)
            labels = self.kmeans.predict(features)
            return torch.from_numpy(labels.astype(np.int64))
    
    def save(self, path: str | Path) -> None:
        """Save the fitted K-means model."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "n_clusters": self.n_clusters,
                "feature_dim": self.feature_dim,
                "hop_length": self.hop_length,
                "sample_rate": self.sample_rate,
                "kmeans": self.kmeans,
            }, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "ClusterLabelGenerator":
        """Load a fitted K-means model."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        instance = cls(
            n_clusters=data["n_clusters"],
            feature_dim=data["feature_dim"],
            hop_length=data["hop_length"],
            sample_rate=data["sample_rate"],
        )
        instance.kmeans = data["kmeans"]
        instance._fitted = True
        return instance


def align_labels_to_features(
    labels: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """
    Align cluster labels to match the transformer feature sequence length.
    
    The spectrogram features may have different frame rate than the CNN encoder.
    This function interpolates labels to match the target length.
    
    Parameters
    ----------
    labels : torch.Tensor
        Cluster labels of shape (batch, label_frames) or (label_frames,)
    target_length : int
        Target sequence length (from CNN encoder)
    
    Returns
    -------
    torch.Tensor
        Aligned labels of shape (batch, target_length) or (target_length,)
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    batch_size, label_len = labels.shape
    
    if label_len == target_length:
        result = labels
    else:
        # Use nearest neighbor interpolation
        indices = torch.linspace(0, label_len - 1, target_length).long()
        indices = indices.clamp(0, label_len - 1)
        result = labels[:, indices]
    
    if squeeze:
        result = result.squeeze(0)
    
    return result
