"""
PyTorch Dataset for STEAD (STanford EArthquake Dataset).

STEAD contains ~1.2M 3-component seismic waveforms sampled at 100Hz.
Each waveform is 60 seconds (6000 samples) with channels [E, N, Z].
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, Optional
from .utils import (
    normalize_waveform, 
    apply_filter,
    log_compress,
    quantile_normalize,
    robust_zscore,
    peak_normalize,
    mean_subtract,
)

# Available normalization modes
NORM_MODES = Literal[
    "zscore",        # Standard z-score (mean=0, std=1)
    "robust_zscore", # Median and MAD based (robust to outliers)
    "peak",          # Scale to [-1, 1] by global max
    "peak_per_ch",   # Scale to [-1, 1] per channel
    "quantile",      # Quantile-based (robust to outliers)
    "log",           # Log compression: sign(x)*log(1+|x|)
    "mean",          # Subtract mean only (no scaling)
    "none",          # No normalization
]



class STEADDataset(Dataset):
    """
    PyTorch Dataset for STEAD seismic waveforms.
    
    The dataset provides 3-channel seismic waveforms (E, N, Z components)
    suitable for self-supervised learning with HuBERT-style architectures.
    
    Parameters
    ----------
    hdf5_path : str | Path
        Path to the STEAD HDF5 file (e.g., 'merge.hdf5')
    csv_path : str | Path
        Path to the STEAD metadata CSV file (e.g., 'merge.csv')
    trace_category : str | None
        Filter by trace category: 'earthquake_local', 'noise', or None for all
    channel : str
        Which channel(s) to use: 'Z' (vertical only), 'all' (E, N, Z), or 'ENZ'
    
    Normalization options (applied in order):
    subtract_mean : bool
        Subtract the mean from each channel (default: True)
    norm_by_std : bool
        Divide by standard deviation (default: True, gives z-score normalization)
    norm_by_max : bool
        Divide by the absolute maximum value (default: False)
    signed_sqrt : bool
        Apply signed power transform: sign(x) * |x|^factor (default: False)
    signed_sqrt_factor : float
        Exponent for signed power transform, 0.5 = sqrt (default: 0.5)
    
    Filtering options:
    highpass_freq : float | None
        High-pass filter cutoff in Hz (default: None)
    lowpass_freq : float | None
        Low-pass filter cutoff in Hz (default: None)
    filter_order : int
        Butterworth filter order (default: 4)
    
    Other options:
    max_samples : int | None
        Limit the number of samples (useful for debugging)
    min_magnitude : float | None
        Minimum earthquake magnitude filter
    max_distance_km : float | None
        Maximum source distance filter
    transform : callable | None
        Optional transform to apply to waveforms
    
    Examples
    --------
    >>> # Default: z-score normalization (subtract mean, divide by std)
    >>> dataset = STEADDataset(hdf5_path, csv_path)
    
    >>> # Scale to [-1, 1] by absolute max
    >>> dataset = STEADDataset(hdf5_path, csv_path, norm_by_std=False, norm_by_max=True)
    
    >>> # With bandpass filter (1-40 Hz)
    >>> dataset = STEADDataset(hdf5_path, csv_path, highpass_freq=1.0, lowpass_freq=40.0)
    
    >>> # Signed sqrt for compressing dynamic range
    >>> dataset = STEADDataset(hdf5_path, csv_path, signed_sqrt=True, norm_by_std=False)
    
    >>> # No normalization
    >>> dataset = STEADDataset(hdf5_path, csv_path, subtract_mean=False, norm_by_std=False)
    """
    
    SAMPLE_RATE = 100  # Hz
    WAVEFORM_LENGTH = 6000  # 60 seconds at 100 Hz
    NUM_CHANNELS = 3  # E, N, Z
    
    def __init__(
        self,
        hdf5_path: str | Path,
        csv_path: str | Path,
        trace_category: Literal["earthquake_local", "noise"] | None = None,
        channel: Literal["Z", "all", "ENZ"] = "Z",
        # Normalization
        norm_mode: NORM_MODES = "zscore",
        # Filter options
        highpass_freq: Optional[float] = None,
        lowpass_freq: Optional[float] = None,
        filter_order: int = 4,
        # Data selection
        max_samples: int | None = None,
        min_magnitude: float | None = None,
        max_distance_km: float | None = None,
        transform=None,
        # Phase picking specific
        return_phase_labels: bool = False,
        label_sigma: float = 10.0,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.csv_path = Path(csv_path)
        self.channel = channel
        self.trace_category = trace_category
        self.transform = transform
        
        self.return_phase_labels = return_phase_labels
        self.label_sigma = label_sigma
        
        # Normalization mode
        self.norm_mode = norm_mode
        
        # Filter settings
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.filter_order = filter_order
        
        self.metadata = self._load_metadata(
            trace_category=trace_category,
            min_magnitude=min_magnitude,
            max_distance_km=max_distance_km,
            max_samples=max_samples,
        )
        
        self.trace_names = self.metadata["trace_name"].tolist()
        self._hdf5_file = None
    
    def _load_metadata(
        self,
        trace_category: str | None,
        min_magnitude: float | None,
        max_distance_km: float | None,
        max_samples: int | None,
    ) -> pd.DataFrame:
        """Load and filter metadata CSV."""
        df = pd.read_csv(self.csv_path, low_memory=False)
        
        if trace_category is not None:
            df = df[df["trace_category"] == trace_category]
        
        if min_magnitude is not None:
            mask = (df["trace_category"] == "noise") | (
                df["source_magnitude"] >= min_magnitude
            )
            df = df[mask]
        
        if max_distance_km is not None:
            mask = (df["trace_category"] == "noise") | (
                df["source_distance_km"] <= max_distance_km
            )
            df = df[mask]
        
        if max_samples is not None:
            df = df.head(max_samples)
        
        return df.reset_index(drop=True)
    
    @property
    def hdf5_file(self) -> h5py.File:
        """Lazy-load HDF5 file (required for multiprocessing)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file
    
    def __len__(self) -> int:
        return len(self.trace_names)
    
    def __repr__(self) -> str:
        # Filter info
        filter_parts = []
        if self.highpass_freq:
            filter_parts.append(f"hp={self.highpass_freq}Hz")
        if self.lowpass_freq:
            filter_parts.append(f"lp={self.lowpass_freq}Hz")
        filter_str = ", ".join(filter_parts) if filter_parts else "none"
        
        # Metadata stats
        n_eq = (self.metadata["trace_category"] == "earthquake_local").sum()
        n_noise = (self.metadata["trace_category"] == "noise").sum()
        
        lines = [
            f"STEADDataset(",
            f"  samples: {len(self)} (earthquakes={n_eq}, noise={n_noise})",
            f"  channel: {self.channel!r}, category: {self.trace_category!r}",
            f"  normalization: {self.norm_mode}, filter: {filter_str}",
        ]
        
        # Add magnitude/distance range for earthquake data
        eq_mask = self.metadata["trace_category"] == "earthquake_local"
        if eq_mask.any():
            eq_data = self.metadata[eq_mask]
            mag_min, mag_max = eq_data["source_magnitude"].min(), eq_data["source_magnitude"].max()
            dist_min, dist_max = eq_data["source_distance_km"].min(), eq_data["source_distance_km"].max()
            lines.append(f"  magnitude: [{mag_min:.1f}, {mag_max:.1f}], distance: [{dist_min:.0f}, {dist_max:.0f}] km")
        
        lines.append(")")
        return "\n".join(lines)
    
    def __getitem__(self, idx: int) -> dict:
        trace_name = self.trace_names[idx]
        dataset = self.hdf5_file.get(f"data/{trace_name}")
        
        if dataset is None:
            raise KeyError(f"Trace '{trace_name}' not found in HDF5 file")
        
        waveform = np.array(dataset, dtype=np.float32)
        
        if self.channel == "Z":
            waveform = waveform[:, 2:3]  # Keep dimension for consistency
        elif self.channel in ("all", "ENZ"):
            pass  # Keep all channels
        
        waveform = waveform.T  # Shape: (channels, samples)
        
        # Apply filtering (before normalization)
        if self.highpass_freq is not None or self.lowpass_freq is not None:
            waveform = apply_filter(
                waveform,
                highpass_freq=self.highpass_freq,
                lowpass_freq=self.lowpass_freq,
                filter_order=self.filter_order,
                sample_rate=self.SAMPLE_RATE,
            )
        
        # Compute amplitude statistics BEFORE normalization (for magnitude prediction)
        # These preserve information that would be lost by z-score normalization
        eps = 1e-10
        amplitude_stats = {
            "log_max_amp": float(np.log10(np.abs(waveform).max() + eps)),
            "log_std": float(np.log10(waveform.std() + eps)),
            "log_energy": float(np.log10((waveform ** 2).sum() + eps)),
            "raw_max_amp": float(np.abs(waveform).max()),
        }
        
        # Apply normalization based on norm_mode
        if self.norm_mode == "zscore":
            waveform = normalize_waveform(waveform, subtract_mean=True, norm_by_std=True)
        elif self.norm_mode == "robust_zscore":
            waveform = robust_zscore(waveform)
        elif self.norm_mode == "peak":
            waveform = peak_normalize(waveform, per_channel=False)
        elif self.norm_mode == "peak_per_ch":
            waveform = peak_normalize(waveform, per_channel=True)
        elif self.norm_mode == "quantile":
            waveform = quantile_normalize(waveform)
        elif self.norm_mode == "log":
            waveform = log_compress(waveform)
        elif self.norm_mode == "mean":
            waveform = mean_subtract(waveform)
        elif self.norm_mode == "none":
            pass  # No normalization
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")
        
        waveform = torch.from_numpy(waveform.astype(np.float32))
        
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        metadata = self.metadata.iloc[idx]
        
        p_arrival = _safe_int(metadata.get("p_arrival_sample"))
        s_arrival = _safe_int(metadata.get("s_arrival_sample"))
        
        result = {
            "waveform": waveform,
            "trace_name": trace_name,
            "trace_category": metadata["trace_category"],
            "p_arrival_sample": p_arrival,
            "s_arrival_sample": s_arrival,
            "source_magnitude": _safe_float(metadata.get("source_magnitude")),
            "source_distance_km": _safe_float(metadata.get("source_distance_km")),
            "source_depth_km": _safe_float(metadata.get("source_depth_km")),
            "amplitude_stats": amplitude_stats,
        }
        
        if self.return_phase_labels:
            phase_labels = generate_phase_labels(
                seq_len=waveform.shape[-1],
                p_sample=p_arrival,
                s_sample=s_arrival,
                sigma=self.label_sigma,
            )
            result["phase_labels"] = torch.from_numpy(phase_labels)
            
        return result
    
    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()
    
    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self),
            "earthquakes": (self.metadata["trace_category"] == "earthquake_local").sum(),
            "noise": (self.metadata["trace_category"] == "noise").sum(),
        }
        
        eq_mask = self.metadata["trace_category"] == "earthquake_local"
        if eq_mask.any():
            eq_data = self.metadata[eq_mask]
            stats["magnitude_range"] = (
                eq_data["source_magnitude"].min(),
                eq_data["source_magnitude"].max(),
            )
            stats["distance_range_km"] = (
                eq_data["source_distance_km"].min(),
                eq_data["source_distance_km"].max(),
            )
        
        return stats
    
    def get_raw_waveform(self, idx: int) -> tuple[np.ndarray, dict]:
        """
        Get raw (unnormalized, unfiltered) waveform and full HDF5 attributes.
        
        Parameters
        ----------
        idx : int
            Sample index
        
        Returns
        -------
        tuple
            (waveform array of shape (samples, 3), attributes dict)
        """
        trace_name = self.trace_names[idx]
        dataset = self.hdf5_file.get(f"data/{trace_name}")
        
        if dataset is None:
            raise KeyError(f"Trace '{trace_name}' not found in HDF5 file")
        
        waveform = np.array(dataset, dtype=np.float32)
        attrs = dict(dataset.attrs)
        
        return waveform, attrs
    
    def to_obspy_stream(self, idx: int):
        """
        Convert a sample to an ObsPy Stream object.
        
        Parameters
        ----------
        idx : int
            Sample index
        
        Returns
        -------
        obspy.Stream
            Stream with 3 traces (E, N, Z components)
        """
        import obspy
        from obspy import UTCDateTime
        
        waveform, attrs = self.get_raw_waveform(idx)
        metadata = self.metadata.iloc[idx]
        
        traces = []
        channel_suffixes = ['E', 'N', 'Z']
        
        for i, suffix in enumerate(channel_suffixes):
            tr = obspy.Trace(data=waveform[:, i])
            tr.stats.sampling_rate = self.SAMPLE_RATE
            tr.stats.delta = 1.0 / self.SAMPLE_RATE
            tr.stats.network = str(metadata.get("network_code", ""))
            tr.stats.station = str(metadata.get("receiver_code", ""))
            tr.stats.channel = str(metadata.get("receiver_type", "HH")) + suffix
            
            start_time = metadata.get("trace_start_time")
            if pd.notna(start_time):
                tr.stats.starttime = UTCDateTime(str(start_time))
            
            traces.append(tr)
        
        return obspy.Stream(traces)


def _safe_int(value) -> int | None:
    """Safely convert to int, handling NaN values."""
    if pd.isna(value):
        return None
    return int(value)


def _safe_float(value) -> float | None:
    """Safely convert to float, handling NaN values."""
    if pd.isna(value):
        return None
    return float(value)


def generate_phase_labels(
    seq_len: int, 
    p_sample: int | None, 
    s_sample: int | None, 
    sigma: float = 10.0
) -> np.ndarray:
    """
    Generate segmentation ground truth for phase picking.
    Returns array of shape (3, seq_len) for [Noise, P, S] probabilities.
    P and S arrivals are represented as Gaussian distributions.
    """
    labels = np.zeros((3, seq_len), dtype=np.float32)
    t = np.arange(seq_len)
    
    if p_sample is not None and p_sample >= 0 and not pd.isna(p_sample):
        labels[1, :] = np.exp(-((t - p_sample) ** 2) / (2 * sigma ** 2))
        
    if s_sample is not None and s_sample >= 0 and not pd.isna(s_sample):
        labels[2, :] = np.exp(-((t - s_sample) ** 2) / (2 * sigma ** 2))
        
    # Noise is 1 minus the sum of P and S (clipped to [0, 1])
    labels[0, :] = 1.0 - np.clip(labels[1, :] + labels[2, :], 0.0, 1.0)
    
    return labels


class STEADCollator:
    """
    Collator for batching STEAD waveforms.
    
    Handles variable-length sequences and creates attention masks
    for transformer models.
    """
    
    def __init__(self, pad_to_length: int | None = None, return_labels: bool = False):
        self.pad_to_length = pad_to_length
        self.return_labels = return_labels
    
    def __call__(self, batch: list[dict]) -> dict:
        waveforms = torch.stack([item["waveform"] for item in batch])
        
        result = {
            "input_values": waveforms,
            "attention_mask": torch.ones(
                waveforms.shape[0], waveforms.shape[-1], dtype=torch.long
            ),
        }
        
        if "phase_labels" in batch[0]:
            result["labels"] = torch.stack([item["phase_labels"] for item in batch])
            
        if self.return_labels:
            result["trace_category"] = [item["trace_category"] for item in batch]
            result["trace_name"] = [item["trace_name"] for item in batch]
        
        # Include distance for adaptive masking (None for noise samples)
        result["source_distance_km"] = torch.tensor([
            item.get("source_distance_km") or -1.0 for item in batch
        ], dtype=torch.float32)
        
        return result
