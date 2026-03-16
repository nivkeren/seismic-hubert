"""
PyTorch Dataset wrapper for SeisBench benchmark datasets.

This module provides a unified interface to one or more SeisBench datasets,
applying the same preprocessing (normalization, filtering) as STEADDataset.

Supported datasets: ETHZ, INSTANCE, LENDB, GEOFON, IQUIQUE, OBS, PNW, SCEDC, STEAD, etc.
Full list: https://seisbench.readthedocs.io/en/stable/pages/data/benchmark_datasets.html
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, Optional, Any, Union
import warnings

from data.utils import (
    normalize_waveform,
    apply_filter,
    log_compress,
    quantile_normalize,
    robust_zscore,
    peak_normalize,
    mean_subtract,
)

# Available normalization modes (same as STEADDataset)
NORM_MODES = Literal[
    "zscore",
    "robust_zscore",
    "peak",
    "peak_per_ch",
    "quantile",
    "log",
    "mean",
    "none",
]

# Standard metadata columns we extract from all datasets
STANDARD_COLUMNS = [
    "trace_name",
    "trace_category",
    "p_arrival_sample",
    "s_arrival_sample",
    "source_magnitude",
    "source_distance_km",
    "source_depth_km",
    "source_latitude",
    "source_longitude",
    "station_code",
    "network_code",
    "split",
    "trace_sampling_rate_hz",
    "trace_npts",
]

# Mapping from various SeisBench column names to our standard names
COLUMN_ALIASES = {
    "trace_name": ["trace_name"],
    "trace_category": ["trace_category", "trace_type"],
    "p_arrival_sample": ["trace_p_arrival_sample", "p_arrival_sample", "trace_p_pick"],
    "s_arrival_sample": ["trace_s_arrival_sample", "s_arrival_sample", "trace_s_pick"],
    "source_magnitude": ["source_magnitude", "source_magnitude_value"],
    "source_distance_km": ["path_ep_distance_km", "source_distance_km", "path_distance_km"],
    "source_depth_km": ["source_depth_km"],
    "source_latitude": ["source_latitude_deg", "source_latitude"],
    "source_longitude": ["source_longitude_deg", "source_longitude"],
    "station_code": ["station_code", "receiver_code"],
    "network_code": ["station_network_code", "network_code"],
    "split": ["split"],
    "trace_sampling_rate_hz": ["trace_sampling_rate_hz", "sampling_rate"],
    "trace_npts": ["trace_npts", "trace_nsamples", "npts"],
}

# SeisBench dataset classes and their info
SEISBENCH_DATASETS = {
    "stead": {"class": "STEAD", "sample_rate": 100, "manual": True},
    "ethz": {"class": "ETHZ", "sample_rate": 100, "manual": False},
    "instance": {"class": "InstanceCounts", "sample_rate": 100, "manual": False},
    "instance_gm": {"class": "InstanceGM", "sample_rate": 100, "manual": False},
    "instance_noise": {"class": "InstanceNoise", "sample_rate": 100, "manual": False},
    "instance_combined": {"class": "InstanceCountsCombined", "sample_rate": 100, "manual": False},
    "lendb": {"class": "LENDB", "sample_rate": 100, "manual": False},
    "geofon": {"class": "GEOFON", "sample_rate": 100, "manual": False},
    "iquique": {"class": "Iquique", "sample_rate": 100, "manual": False},
    "neic": {"class": "NEIC", "sample_rate": 100, "manual": False},
    "mlaapde": {"class": "MLAAPDE", "sample_rate": 100, "manual": False},
    "obs": {"class": "OBS", "sample_rate": 100, "manual": False},
    "obst2024": {"class": "OBST2024", "sample_rate": 100, "manual": False},
    "pnw": {"class": "PNW", "sample_rate": 100, "manual": False},
    "pnw_noise": {"class": "PNWNoise", "sample_rate": 100, "manual": False},
    "pnw_exotic": {"class": "PNWExotic", "sample_rate": 100, "manual": False},
    "pnw_accelerometers": {"class": "PNWAccelerometers", "sample_rate": 100, "manual": False},
    "scedc": {"class": "SCEDC", "sample_rate": 100, "manual": False},
    "txed": {"class": "TXED", "sample_rate": 100, "manual": False},
    "vcseis": {"class": "VCSEIS", "sample_rate": 100, "manual": False},
    "aq2009": {"class": "AQ2009Counts", "sample_rate": 100, "manual": False},
    "aq2009_gm": {"class": "AQ2009GM", "sample_rate": 100, "manual": False},
    "ceed": {"class": "CEED", "sample_rate": 100, "manual": False},
    "crew": {"class": "CREW", "sample_rate": 100, "manual": False},
    "cwa": {"class": "CWA", "sample_rate": 100, "manual": False},
    "pisdl": {"class": "PiSDL", "sample_rate": 100, "manual": False},
    "isc_ehb_depth": {"class": "ISC_EHB_DepthPhases", "sample_rate": 100, "manual": False},
    "lfe_cascadia": {"class": "LFEStacksCascadiaBostock2015", "sample_rate": 100, "manual": False},
    "lfe_mexico": {"class": "LFEStacksMexicoFrank2014", "sample_rate": 100, "manual": False},
    "lfe_san_andreas": {"class": "LFEStacksSanAndreasShelly2017", "sample_rate": 100, "manual": False},
}


def _load_seisbench_dataset(name: str, cache_root: Optional[str] = None, **kwargs):
    """Load a single SeisBench dataset by name."""
    import seisbench.data as sbd
    
    name = name.lower()
    if name not in SEISBENCH_DATASETS:
        available = ", ".join(sorted(SEISBENCH_DATASETS.keys()))
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    info = SEISBENCH_DATASETS[name]
    class_name = info["class"]
    
    if not hasattr(sbd, class_name):
        raise ValueError(f"SeisBench class {class_name} not found. Update seisbench?")
    
    DatasetClass = getattr(sbd, class_name)
    
    init_kwargs = {}
    if cache_root:
        init_kwargs["cache_root"] = cache_root
    init_kwargs.update(kwargs)
    
    return DatasetClass(**init_kwargs)


def _find_column(metadata: pd.DataFrame, standard_name: str) -> Optional[str]:
    """Find the actual column name in metadata for a standard field."""
    if standard_name not in COLUMN_ALIASES:
        return None
    
    for col in COLUMN_ALIASES[standard_name]:
        if col in metadata.columns:
            return col
    return None


def _normalize_metadata(
    metadata: pd.DataFrame, 
    dataset_name: str,
    default_sampling_rate: Optional[float] = None,
    default_npts: Optional[int] = None,
) -> pd.DataFrame:
    """Normalize metadata columns to standard names."""
    normalized = pd.DataFrame(index=metadata.index)
    
    for std_col in STANDARD_COLUMNS:
        src_col = _find_column(metadata, std_col)
        if src_col:
            normalized[std_col] = metadata[src_col]
        else:
            normalized[std_col] = None
    
    # Add dataset source column
    normalized["dataset"] = dataset_name
    
    # Generate trace_name if missing
    if normalized["trace_name"].isna().all():
        normalized["trace_name"] = [f"{dataset_name}_{i}" for i in range(len(normalized))]
    
    # Fill in default sampling rate if not per-trace
    if normalized["trace_sampling_rate_hz"].isna().all() and default_sampling_rate is not None:
        normalized["trace_sampling_rate_hz"] = default_sampling_rate
    
    # Fill in default npts if not per-trace
    if normalized["trace_npts"].isna().all() and default_npts is not None:
        normalized["trace_npts"] = default_npts
    
    return normalized


class SeismicBenchDataset(Dataset):
    """
    Unified PyTorch Dataset for one or more SeisBench benchmark datasets.
    
    Loads and combines multiple datasets with unified metadata,
    applying consistent preprocessing (normalization, filtering).
    
    Parameters
    ----------
    datasets : str or list of str
        Dataset name(s) to load (e.g., 'ethz' or ['ethz', 'lendb', 'iquique'])
    cache_root : str | Path, optional
        Cache directory for downloaded data
    channel : str
        Which channel(s) to use: 'Z' (vertical only) or 'all' (all channels)
    split : str, optional
        Data split: 'train', 'dev', 'test', or None for all
    norm_mode : str
        Normalization mode (same options as STEADDataset)
    highpass_freq : float, optional
        High-pass filter cutoff in Hz
    lowpass_freq : float, optional
        Low-pass filter cutoff in Hz
    filter_order : int
        Butterworth filter order
    max_samples : int, optional
        Limit total samples (applied after combining datasets)
    max_samples_per_dataset : int, optional
        Limit samples per dataset (before combining)
    min_magnitude : float, optional
        Minimum earthquake magnitude filter
    max_distance_km : float, optional
        Maximum source distance filter
    min_sampling_rate : float, optional
        Minimum waveform sampling rate in Hz (filter out lower rates)
    max_sampling_rate : float, optional
        Maximum waveform sampling rate in Hz (filter out higher rates)
    min_trace_length : int, optional
        Minimum waveform length in samples (filter out shorter traces)
    max_trace_length : int, optional
        Maximum waveform length in samples (filter out longer traces)
    target_length : int, optional
        Resample waveforms to this length (default: 6000 = 60s at 100Hz)
    balance_datasets : bool
        If True, balance samples across datasets (default: False)
    
    Examples
    --------
    >>> # Single dataset
    >>> dataset = SeismicBenchDataset("ethz")
    
    >>> # Multiple datasets combined
    >>> dataset = SeismicBenchDataset(
    ...     ["ethz", "lendb", "iquique"],
    ...     split="train",
    ...     norm_mode="zscore",
    ...     highpass_freq=1.0,
    ... )
    
    >>> # With sample limits per dataset
    >>> dataset = SeismicBenchDataset(
    ...     ["ethz", "instance", "scedc"],
    ...     max_samples_per_dataset=10000,  # 10k from each
    ... )
    """
    
    def __init__(
        self,
        datasets: Union[str, list[str]],
        cache_root: Optional[str | Path] = None,
        channel: Literal["Z", "all", "ENZ"] = "Z",
        split: Optional[Literal["train", "dev", "test"]] = None,
        # Normalization
        norm_mode: NORM_MODES = "zscore",
        # Filter options
        highpass_freq: Optional[float] = None,
        lowpass_freq: Optional[float] = None,
        filter_order: int = 4,
        # Data selection
        max_samples: Optional[int] = None,
        max_samples_per_dataset: Optional[int] = None,
        min_magnitude: Optional[float] = None,
        max_distance_km: Optional[float] = None,
        # Waveform metadata filters
        min_sampling_rate: Optional[float] = None,
        max_sampling_rate: Optional[float] = None,
        min_trace_length: Optional[int] = None,
        max_trace_length: Optional[int] = None,
        # Waveform options
        target_length: int = 6000,
        sample_rate: int = 100,
        # Balance
        balance_datasets: bool = False,
        transform=None,
        # Additional SeisBench kwargs per dataset
        dataset_kwargs: Optional[dict[str, dict]] = None,
    ):
        # Normalize input to list
        if isinstance(datasets, str):
            datasets = [datasets]
        
        self.dataset_names = [d.lower() for d in datasets]
        self.channel = channel
        self.norm_mode = norm_mode
        self.highpass_freq = highpass_freq
        self.lowpass_freq = lowpass_freq
        self.filter_order = filter_order
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.transform = transform
        self.split = split
        
        cache_root_str = str(cache_root) if cache_root else None
        dataset_kwargs = dataset_kwargs or {}
        
        # Load all datasets and build unified metadata
        self._sb_datasets = {}  # name -> seisbench dataset
        metadata_parts = []
        
        for name in self.dataset_names:
            try:
                kwargs = dataset_kwargs.get(name, {})
                sb_ds = _load_seisbench_dataset(name, cache_root=cache_root_str, **kwargs)
                self._sb_datasets[name] = sb_ds
                
                # Get default sampling rate and npts from dataset
                default_sr = None
                default_npts = None
                if hasattr(sb_ds, "sampling_rate"):
                    default_sr = sb_ds.sampling_rate
                elif name in SEISBENCH_DATASETS:
                    default_sr = SEISBENCH_DATASETS[name].get("sample_rate", 100)
                
                # Try to get default trace length from data format
                if hasattr(sb_ds, "data_format") and sb_ds.data_format:
                    default_npts = sb_ds.data_format.get("trace_npts")
                
                # Normalize metadata
                ds_metadata = _normalize_metadata(
                    sb_ds.metadata.copy(), 
                    name,
                    default_sampling_rate=default_sr,
                    default_npts=default_npts,
                )
                
                # Add original index for waveform retrieval
                ds_metadata["_original_idx"] = np.arange(len(ds_metadata))
                ds_metadata["_dataset"] = name
                
                metadata_parts.append(ds_metadata)
                print(f"Loaded {name}: {len(ds_metadata)} samples")
                
            except Exception as e:
                warnings.warn(f"Failed to load dataset {name}: {e}")
        
        if not metadata_parts:
            raise ValueError("No datasets could be loaded")
        
        # Combine metadata
        self.metadata = pd.concat(metadata_parts, ignore_index=True)
        print(f"Combined: {len(self.metadata)} total samples")
        
        # Apply filters
        self._apply_filters(
            split=split,
            min_magnitude=min_magnitude,
            max_distance_km=max_distance_km,
            min_sampling_rate=min_sampling_rate,
            max_sampling_rate=max_sampling_rate,
            min_trace_length=min_trace_length,
            max_trace_length=max_trace_length,
            max_samples_per_dataset=max_samples_per_dataset,
            balance_datasets=balance_datasets,
        )
        
        # Apply global max_samples limit
        if max_samples is not None and len(self.metadata) > max_samples:
            self.metadata = self.metadata.sample(n=max_samples, random_state=42)
            self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"After filtering: {len(self.metadata)} samples")
    
    def _apply_filters(
        self,
        split: Optional[str],
        min_magnitude: Optional[float],
        max_distance_km: Optional[float],
        min_sampling_rate: Optional[float],
        max_sampling_rate: Optional[float],
        min_trace_length: Optional[int],
        max_trace_length: Optional[int],
        max_samples_per_dataset: Optional[int],
        balance_datasets: bool,
    ):
        """Apply filters to the combined metadata."""
        mask = np.ones(len(self.metadata), dtype=bool)
        
        # Filter by split
        if split is not None:
            split_mask = self.metadata["split"].isna() | (self.metadata["split"] == split)
            mask &= split_mask
        
        # Filter by magnitude
        if min_magnitude is not None:
            mag_mask = self.metadata["source_magnitude"].isna() | (
                self.metadata["source_magnitude"] >= min_magnitude
            )
            mask &= mag_mask
        
        # Filter by distance
        if max_distance_km is not None:
            dist_mask = self.metadata["source_distance_km"].isna() | (
                self.metadata["source_distance_km"] <= max_distance_km
            )
            mask &= dist_mask
        
        # Filter by sampling rate
        if min_sampling_rate is not None:
            sr_col = self.metadata["trace_sampling_rate_hz"]
            sr_mask = sr_col.isna() | (sr_col >= min_sampling_rate)
            mask &= sr_mask
        
        if max_sampling_rate is not None:
            sr_col = self.metadata["trace_sampling_rate_hz"]
            sr_mask = sr_col.isna() | (sr_col <= max_sampling_rate)
            mask &= sr_mask
        
        # Filter by trace length (number of samples)
        if min_trace_length is not None:
            npts_col = self.metadata["trace_npts"]
            npts_mask = npts_col.isna() | (npts_col >= min_trace_length)
            mask &= npts_mask
        
        if max_trace_length is not None:
            npts_col = self.metadata["trace_npts"]
            npts_mask = npts_col.isna() | (npts_col <= max_trace_length)
            mask &= npts_mask
        
        self.metadata = self.metadata[mask].reset_index(drop=True)
        
        # Apply per-dataset limits
        if max_samples_per_dataset is not None or balance_datasets:
            if balance_datasets and max_samples_per_dataset is None:
                # Balance to smallest dataset size
                sizes = self.metadata.groupby("_dataset").size()
                max_samples_per_dataset = sizes.min()
            
            if max_samples_per_dataset:
                parts = []
                for name in self.dataset_names:
                    ds_data = self.metadata[self.metadata["_dataset"] == name]
                    if len(ds_data) > max_samples_per_dataset:
                        ds_data = ds_data.sample(n=max_samples_per_dataset, random_state=42)
                    parts.append(ds_data)
                
                self.metadata = pd.concat(parts, ignore_index=True)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __repr__(self) -> str:
        filter_parts = []
        if self.highpass_freq:
            filter_parts.append(f"hp={self.highpass_freq}Hz")
        if self.lowpass_freq:
            filter_parts.append(f"lp={self.lowpass_freq}Hz")
        filter_str = ", ".join(filter_parts) if filter_parts else "none"
        
        # Count per dataset
        counts = self.metadata["_dataset"].value_counts()
        
        lines = [
            f"SeismicBenchDataset(",
            f"  datasets: {self.dataset_names}",
            f"  total_samples: {len(self)}",
        ]
        for name in self.dataset_names:
            n = counts.get(name, 0)
            lines.append(f"    - {name}: {n}")
        
        lines.extend([
            f"  channel: {self.channel!r}, split: {self.split!r}",
            f"  normalization: {self.norm_mode}, filter: {filter_str}",
            f"  target_length: {self.target_length}, sample_rate: {self.sample_rate}Hz",
            ")",
        ])
        return "\n".join(lines)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        dataset_name = row["_dataset"]
        orig_idx = int(row["_original_idx"])
        
        # Get waveform from the appropriate SeisBench dataset
        sb_dataset = self._sb_datasets[dataset_name]
        waveform = sb_dataset.get_waveforms(orig_idx)
        
        # Handle different array shapes
        # SeisBench returns (channels, samples) or (samples, channels)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        elif waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T
        
        waveform = waveform.astype(np.float32)
        
        # Select channels
        n_channels = waveform.shape[0]
        if self.channel == "Z":
            if n_channels >= 3:
                waveform = waveform[2:3, :]  # Z is index 2 in ZNE order
            else:
                waveform = waveform[0:1, :]
        
        # Resample if needed
        if self.target_length is not None and waveform.shape[1] != self.target_length:
            from scipy.signal import resample
            waveform = resample(waveform, self.target_length, axis=1)
        
        # Apply filtering
        if self.highpass_freq is not None or self.lowpass_freq is not None:
            waveform = apply_filter(
                waveform,
                highpass_freq=self.highpass_freq,
                lowpass_freq=self.lowpass_freq,
                filter_order=self.filter_order,
                sample_rate=self.sample_rate,
            )
        
        # Compute amplitude statistics BEFORE normalization
        eps = 1e-10
        amplitude_stats = {
            "log_max_amp": float(np.log10(np.abs(waveform).max() + eps)),
            "log_std": float(np.log10(waveform.std() + eps)),
            "log_energy": float(np.log10((waveform ** 2).sum() + eps)),
            "raw_max_amp": float(np.abs(waveform).max()),
        }
        
        # Apply normalization
        waveform = self._apply_normalization(waveform)
        
        waveform = torch.from_numpy(waveform.astype(np.float32))
        
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        return {
            "waveform": waveform,
            "trace_name": str(row["trace_name"]),
            "trace_category": str(row["trace_category"]) if pd.notna(row["trace_category"]) else "unknown",
            "p_arrival_sample": _safe_int(row["p_arrival_sample"]),
            "s_arrival_sample": _safe_int(row["s_arrival_sample"]),
            "source_magnitude": _safe_float(row["source_magnitude"]),
            "source_distance_km": _safe_float(row["source_distance_km"]),
            "source_depth_km": _safe_float(row["source_depth_km"]),
            "amplitude_stats": amplitude_stats,
            "dataset": dataset_name,
        }
    
    def _apply_normalization(self, waveform: np.ndarray) -> np.ndarray:
        """Apply normalization based on norm_mode."""
        if self.norm_mode == "zscore":
            return normalize_waveform(waveform, subtract_mean=True, norm_by_std=True)
        elif self.norm_mode == "robust_zscore":
            return robust_zscore(waveform)
        elif self.norm_mode == "peak":
            return peak_normalize(waveform, per_channel=False)
        elif self.norm_mode == "peak_per_ch":
            return peak_normalize(waveform, per_channel=True)
        elif self.norm_mode == "quantile":
            return quantile_normalize(waveform)
        elif self.norm_mode == "log":
            return log_compress(waveform)
        elif self.norm_mode == "mean":
            return mean_subtract(waveform)
        elif self.norm_mode == "none":
            return waveform
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")
    
    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self),
            "datasets": self.dataset_names,
            "split": self.split,
            "samples_per_dataset": self.metadata["_dataset"].value_counts().to_dict(),
        }
        
        # Category distribution
        if "trace_category" in self.metadata.columns:
            cat_counts = self.metadata["trace_category"].value_counts()
            stats["categories"] = cat_counts.to_dict()
        
        # Magnitude range
        mags = self.metadata["source_magnitude"].dropna()
        if len(mags) > 0:
            stats["magnitude_range"] = (float(mags.min()), float(mags.max()))
        
        # Distance range
        dists = self.metadata["source_distance_km"].dropna()
        if len(dists) > 0:
            stats["distance_range_km"] = (float(dists.min()), float(dists.max()))
        
        # Sampling rate distribution
        sr = self.metadata["trace_sampling_rate_hz"].dropna()
        if len(sr) > 0:
            unique_sr = sr.unique()
            if len(unique_sr) == 1:
                stats["sampling_rate_hz"] = float(unique_sr[0])
            else:
                stats["sampling_rate_hz_range"] = (float(sr.min()), float(sr.max()))
                stats["sampling_rate_hz_unique"] = sorted([float(x) for x in unique_sr])
        
        # Trace length distribution
        npts = self.metadata["trace_npts"].dropna()
        if len(npts) > 0:
            unique_npts = npts.unique()
            if len(unique_npts) == 1:
                stats["trace_npts"] = int(unique_npts[0])
            else:
                stats["trace_npts_range"] = (int(npts.min()), int(npts.max()))
        
        return stats
    
    def get_dataset_weights(self) -> torch.Tensor:
        """
        Get per-sample weights for balanced sampling across datasets.
        
        Use with WeightedRandomSampler for balanced training.
        
        Returns
        -------
        torch.Tensor
            Weight for each sample (inverse of dataset frequency)
        """
        counts = self.metadata["_dataset"].value_counts()
        weights = self.metadata["_dataset"].map(lambda x: 1.0 / counts[x])
        return torch.tensor(weights.values, dtype=torch.float32)


def _safe_int(value) -> Optional[int]:
    """Safely convert to int, handling NaN values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> Optional[float]:
    """Safely convert to float, handling NaN values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
