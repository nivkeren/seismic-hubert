"""
K-means clustering for HuBERT-style self-supervised learning targets.

This module provides functionality to:
1. Extract spectrogram features from seismic waveforms
2. Run K-means clustering on these features
3. Generate cluster labels for training targets

Domain-specific features for seismic data:
- Spectrogram features (default): STFT-based frequency content
- STA/LTA features: Short-term/Long-term Average ratio for transient detection
- Multi-channel features: Cross-channel correlation for polarization analysis
- Frequency band energy: Energy in specific seismic frequency bands
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
from typing import Optional, Literal

# Feature extraction modes
FEATURE_MODES = Literal["spectrogram", "stalta", "frequency_bands", "multi_channel", "combined"]


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


def extract_stalta_features(
    waveform: np.ndarray,
    sample_rate: int = 100,
    sta_window: float = 1.0,
    lta_window: float = 10.0,
    hop_length: int = 32,
) -> np.ndarray:
    """
    Extract STA/LTA (Short-Term Average / Long-Term Average) features.
    
    STA/LTA is a classic seismic detection method that captures transient energy
    changes, making it ideal for identifying P-wave onsets and other impulsive signals.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform of shape (channels, samples) or (samples,)
    sample_rate : int
        Sample rate in Hz
    sta_window : float
        Short-term window length in seconds
    lta_window : float
        Long-term window length in seconds
    hop_length : int
        Hop length between frames
    
    Returns
    -------
    np.ndarray
        STA/LTA features of shape (frames, features)
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    
    n_channels = min(waveform.shape[0], 3)
    sta_samples = int(sta_window * sample_rate)
    lta_samples = int(lta_window * sample_rate)
    
    features_list = []
    
    for ch in range(n_channels):
        wav = waveform[ch]
        wav_squared = wav ** 2
        
        # Compute running averages using convolution
        sta_kernel = np.ones(sta_samples) / sta_samples
        lta_kernel = np.ones(lta_samples) / lta_samples
        
        sta = np.convolve(wav_squared, sta_kernel, mode='same')
        lta = np.convolve(wav_squared, lta_kernel, mode='same')
        
        # STA/LTA ratio (with epsilon to avoid division by zero)
        stalta = sta / (lta + 1e-10)
        
        # Also include raw energy as a feature
        energy = np.log(sta + 1e-10)
        
        # Downsample to frame rate
        n_frames = len(wav) // hop_length
        stalta_frames = stalta[::hop_length][:n_frames]
        energy_frames = energy[::hop_length][:n_frames]
        
        features_list.extend([stalta_frames, energy_frames])
    
    # Stack features: (frames, n_channels * 2)
    features = np.stack(features_list, axis=1)
    
    return features.astype(np.float32)


def extract_frequency_band_features(
    waveform: np.ndarray,
    sample_rate: int = 100,
    hop_length: int = 32,
    bands: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Extract energy in specific seismic frequency bands.
    
    Default bands are chosen for typical seismic signals:
    - 0.1-1 Hz: Teleseismic and surface waves
    - 1-5 Hz: Regional earthquakes
    - 5-20 Hz: Local earthquakes
    - 20-40 Hz: High-frequency local events
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform of shape (channels, samples) or (samples,)
    sample_rate : int
        Sample rate in Hz
    hop_length : int
        Hop length between frames
    bands : list of tuples, optional
        Frequency bands as (low_hz, high_hz) tuples
    
    Returns
    -------
    np.ndarray
        Band energy features of shape (frames, n_bands * n_channels)
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    
    if bands is None:
        # Default seismic frequency bands
        bands = [
            (0.1, 1.0),   # Teleseismic
            (1.0, 5.0),   # Regional
            (5.0, 20.0),  # Local
            (20.0, 40.0), # High-frequency
        ]
    
    n_channels = min(waveform.shape[0], 3)
    n_samples = waveform.shape[1]
    n_frames = n_samples // hop_length
    
    features_list = []
    
    for ch in range(n_channels):
        wav = waveform[ch]
        
        for low, high in bands:
            # Skip bands outside Nyquist frequency
            if low >= sample_rate / 2:
                continue
            high = min(high, sample_rate / 2 - 0.1)
            
            # Design bandpass filter
            sos = signal.butter(4, [low, high], btype='band', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, wav)
            
            # Compute envelope using Hilbert transform
            envelope = np.abs(signal.hilbert(filtered))
            log_envelope = np.log(envelope + 1e-10)
            
            # Downsample to frame rate
            band_features = log_envelope[::hop_length][:n_frames]
            features_list.append(band_features)
    
    features = np.stack(features_list, axis=1)
    
    return features.astype(np.float32)


def extract_multichannel_features(
    waveform: np.ndarray,
    sample_rate: int = 100,
    hop_length: int = 32,
    window_samples: int = 100,
) -> np.ndarray:
    """
    Extract multi-channel polarization features for 3-component seismograms.
    
    Computes cross-channel correlations and polarization attributes that capture
    the directional properties of seismic waves (useful for distinguishing P, S,
    and surface waves).
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform of shape (3, samples) for E, N, Z components
    sample_rate : int
        Sample rate in Hz
    hop_length : int
        Hop length between frames
    window_samples : int
        Window size for computing correlations
    
    Returns
    -------
    np.ndarray
        Polarization features of shape (frames, features)
    """
    if waveform.ndim == 1:
        # Single channel - return dummy features
        n_frames = len(waveform) // hop_length
        return np.zeros((n_frames, 6), dtype=np.float32)
    
    if waveform.shape[0] < 3:
        # Pad to 3 channels if needed
        padding = np.zeros((3 - waveform.shape[0], waveform.shape[1]))
        waveform = np.vstack([waveform, padding])
    
    n_samples = waveform.shape[1]
    n_frames = n_samples // hop_length
    
    # Extract E, N, Z components
    E, N, Z = waveform[0], waveform[1], waveform[2]
    
    features_list = []
    
    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        end = min(start + window_samples, n_samples)
        
        e_win = E[start:end]
        n_win = N[start:end]
        z_win = Z[start:end]
        
        # Compute covariance matrix elements
        cov_ee = np.mean(e_win ** 2)
        cov_nn = np.mean(n_win ** 2)
        cov_zz = np.mean(z_win ** 2)
        cov_en = np.mean(e_win * n_win)
        cov_ez = np.mean(e_win * z_win)
        cov_nz = np.mean(n_win * z_win)
        
        # Horizontal to vertical ratio (P vs S wave indicator)
        h_energy = cov_ee + cov_nn
        v_energy = cov_zz
        hv_ratio = np.log((h_energy + 1e-10) / (v_energy + 1e-10))
        
        # Total energy
        total_energy = np.log(h_energy + v_energy + 1e-10)
        
        # Cross-correlations (normalized)
        norm = np.sqrt(cov_ee * cov_nn * cov_zz + 1e-10)
        en_corr = cov_en / (np.sqrt(cov_ee * cov_nn) + 1e-10)
        ez_corr = cov_ez / (np.sqrt(cov_ee * cov_zz) + 1e-10)
        nz_corr = cov_nz / (np.sqrt(cov_nn * cov_zz) + 1e-10)
        
        # Rectilinearity (how linear vs elliptical the particle motion is)
        # Simplified: ratio of dominant to minor eigenvalue
        cov_matrix = np.array([
            [cov_ee, cov_en, cov_ez],
            [cov_en, cov_nn, cov_nz],
            [cov_ez, cov_nz, cov_zz]
        ])
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        rectilinearity = 1 - (eigenvalues[1] + eigenvalues[2]) / (2 * eigenvalues[0] + 1e-10)
        
        features_list.append([
            hv_ratio, total_energy, en_corr, ez_corr, nz_corr, rectilinearity
        ])
    
    features = np.array(features_list, dtype=np.float32)
    
    return features


def extract_combined_features(
    waveform: np.ndarray,
    sample_rate: int = 100,
    hop_length: int = 32,
    include_spectrogram: bool = True,
    include_stalta: bool = True,
    include_frequency_bands: bool = True,
    include_multichannel: bool = True,
    n_fft: int = 64,
    n_mels: int = 32,
) -> np.ndarray:
    """
    Extract combined features from multiple feature types.
    
    This provides the richest representation for K-means clustering by combining
    spectral, temporal, and polarization information.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform
    sample_rate : int
        Sample rate in Hz
    hop_length : int
        Hop length between frames
    include_spectrogram : bool
        Include STFT spectrogram features
    include_stalta : bool
        Include STA/LTA features
    include_frequency_bands : bool
        Include frequency band energy features
    include_multichannel : bool
        Include multi-channel polarization features
    n_fft : int
        FFT size for spectrogram
    n_mels : int
        Number of frequency bins for spectrogram
    
    Returns
    -------
    np.ndarray
        Combined features of shape (frames, total_features)
    """
    feature_parts = []
    
    if include_spectrogram:
        spec_feat = extract_spectrogram_features(
            waveform, sample_rate, n_fft, hop_length, n_mels
        )
        feature_parts.append(spec_feat)
    
    if include_stalta:
        stalta_feat = extract_stalta_features(waveform, sample_rate, hop_length=hop_length)
        feature_parts.append(stalta_feat)
    
    if include_frequency_bands:
        band_feat = extract_frequency_band_features(waveform, sample_rate, hop_length)
        feature_parts.append(band_feat)
    
    if include_multichannel and waveform.ndim > 1 and waveform.shape[0] >= 3:
        mc_feat = extract_multichannel_features(waveform, sample_rate, hop_length)
        feature_parts.append(mc_feat)
    
    if not feature_parts:
        raise ValueError("At least one feature type must be enabled")
    
    # Align frame counts (use minimum)
    min_frames = min(f.shape[0] for f in feature_parts)
    feature_parts = [f[:min_frames] for f in feature_parts]
    
    return np.concatenate(feature_parts, axis=1).astype(np.float32)


class ClusterLabelGenerator:
    """
    Generates cluster labels for HuBERT training.
    
    This class handles:
    1. Extracting features from waveforms (multiple feature types available)
    2. Training K-means clustering
    3. Assigning cluster labels to new waveforms
    
    Feature modes:
    - "spectrogram": STFT-based frequency content (default, similar to audio HuBERT)
    - "stalta": STA/LTA ratio features for transient detection
    - "frequency_bands": Energy in seismic frequency bands
    - "multi_channel": Polarization features for 3-component data
    - "combined": All features concatenated for rich representation
    """
    
    def __init__(
        self,
        n_clusters: int = 100,
        feature_dim: int = 32,
        hop_length: int = 32,
        sample_rate: int = 100,
        feature_mode: FEATURE_MODES = "spectrogram",
        include_stalta: bool = False,
        include_frequency_bands: bool = False,
        include_multichannel: bool = False,
    ):
        """
        Initialize the ClusterLabelGenerator.
        
        Parameters
        ----------
        n_clusters : int
            Number of K-means clusters (target vocabulary size)
        feature_dim : int
            Feature dimension for spectrogram (n_mels)
        hop_length : int
            Hop length between frames in samples
        sample_rate : int
            Sample rate in Hz
        feature_mode : str
            Feature extraction mode: "spectrogram", "stalta", "frequency_bands",
            "multi_channel", or "combined"
        include_stalta : bool
            Include STA/LTA features when feature_mode="combined"
        include_frequency_bands : bool
            Include frequency band features when feature_mode="combined"
        include_multichannel : bool
            Include polarization features when feature_mode="combined"
        """
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.feature_mode = feature_mode
        self.include_stalta = include_stalta
        self.include_frequency_bands = include_frequency_bands
        self.include_multichannel = include_multichannel
        self.kmeans: Optional[MiniBatchKMeans] = None
        self._fitted = False
    
    def extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract features from a single waveform based on feature_mode.
        
        Parameters
        ----------
        waveform : np.ndarray
            Input waveform of shape (channels, samples) or (samples,)
        
        Returns
        -------
        np.ndarray
            Features of shape (frames, features)
        """
        if self.feature_mode == "spectrogram":
            return extract_spectrogram_features(
                waveform,
                sample_rate=self.sample_rate,
                n_fft=self.feature_dim * 2,
                hop_length=self.hop_length,
                n_mels=self.feature_dim,
            )
        elif self.feature_mode == "stalta":
            return extract_stalta_features(
                waveform,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_mode == "frequency_bands":
            return extract_frequency_band_features(
                waveform,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_mode == "multi_channel":
            return extract_multichannel_features(
                waveform,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_mode == "combined":
            return extract_combined_features(
                waveform,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                include_spectrogram=True,
                include_stalta=self.include_stalta,
                include_frequency_bands=self.include_frequency_bands,
                include_multichannel=self.include_multichannel,
                n_fft=self.feature_dim * 2,
                n_mels=self.feature_dim,
            )
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
    
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
                "feature_mode": self.feature_mode,
                "include_stalta": self.include_stalta,
                "include_frequency_bands": self.include_frequency_bands,
                "include_multichannel": self.include_multichannel,
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
            feature_mode=data.get("feature_mode", "spectrogram"),
            include_stalta=data.get("include_stalta", False),
            include_frequency_bands=data.get("include_frequency_bands", False),
            include_multichannel=data.get("include_multichannel", False),
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
