"""
Configuration dataclasses for waveform processing.

This module provides dataclass-based configuration for normalization and filtering.
These can be used as an alternative to flat parameters when more complex
configuration management is needed (e.g., saving/loading experiment configs).

Example usage:
    from data.configs import NormalizationConfig, FilterConfig
    
    norm = NormalizationConfig(subtract_mean=True, norm_by_max=True)
    filt = FilterConfig.bandpass(low=1.0, high=40.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal


@dataclass
class NormalizationConfig:
    """
    Configuration for waveform normalization.
    
    Multiple normalization steps can be combined and are applied in order:
    1. subtract_mean
    2. norm_by_std
    3. norm_by_max
    4. signed_sqrt
    
    Parameters
    ----------
    subtract_mean : bool
        Subtract the mean from each channel
    norm_by_std : bool
        Divide by standard deviation (after mean subtraction if enabled)
    norm_by_max : bool
        Divide by the absolute maximum value
    signed_sqrt : bool
        Apply signed square root: sign(x) * |x|^factor
    signed_sqrt_factor : float
        Exponent for signed power transform (0.5 = sqrt)
    eps : float
        Small epsilon to avoid division by zero
    """
    subtract_mean: bool = True
    norm_by_std: bool = True
    norm_by_max: bool = False
    signed_sqrt: bool = False
    signed_sqrt_factor: float = 0.5
    eps: float = 1e-8
    
    @classmethod
    def none(cls) -> "NormalizationConfig":
        """No normalization."""
        return cls(subtract_mean=False, norm_by_std=False)
    
    @classmethod
    def standard(cls) -> "NormalizationConfig":
        """Standard z-score normalization (mean=0, std=1)."""
        return cls(subtract_mean=True, norm_by_std=True)
    
    @classmethod
    def max_scale(cls) -> "NormalizationConfig":
        """Scale to [-1, 1] by dividing by absolute maximum."""
        return cls(subtract_mean=False, norm_by_std=False, norm_by_max=True)
    
    def apply(self, waveform: np.ndarray) -> np.ndarray:
        """Apply this normalization config to a waveform."""
        return apply_normalization_config(waveform, self)


@dataclass
class FilterConfig:
    """
    Configuration for frequency filtering.
    
    Uses Butterworth filters from scipy.signal.
    
    Parameters
    ----------
    highpass_freq : float | None
        High-pass filter cutoff frequency in Hz (removes frequencies below this)
    lowpass_freq : float | None
        Low-pass filter cutoff frequency in Hz (removes frequencies above this)
    filter_order : int
        Order of the Butterworth filter
    sample_rate : int
        Sample rate of the waveform in Hz
    """
    highpass_freq: Optional[float] = None
    lowpass_freq: Optional[float] = None
    filter_order: int = 4
    sample_rate: int = 100
    
    @classmethod
    def none(cls) -> "FilterConfig":
        """No filtering."""
        return cls()
    
    @classmethod
    def bandpass(cls, low: float, high: float, order: int = 4) -> "FilterConfig":
        """Bandpass filter between low and high frequencies."""
        return cls(highpass_freq=low, lowpass_freq=high, filter_order=order)
    
    @classmethod
    def highpass(cls, freq: float, order: int = 4) -> "FilterConfig":
        """High-pass filter (removes low frequencies)."""
        return cls(highpass_freq=freq, filter_order=order)
    
    @classmethod
    def lowpass(cls, freq: float, order: int = 4) -> "FilterConfig":
        """Low-pass filter (removes high frequencies)."""
        return cls(lowpass_freq=freq, filter_order=order)
    
    def apply(self, waveform: np.ndarray) -> np.ndarray:
        """Apply this filter config to a waveform."""
        return apply_filter_config(waveform, self)


def apply_normalization_config(waveform: np.ndarray, config: NormalizationConfig) -> np.ndarray:
    """
    Apply normalization to waveform using config.
    
    Parameters
    ----------
    waveform : np.ndarray
        Waveform array of shape (channels, samples)
    config : NormalizationConfig
        Normalization configuration
    
    Returns
    -------
    np.ndarray
        Normalized waveform
    """
    if config.subtract_mean:
        mean = waveform.mean(axis=-1, keepdims=True)
        waveform = waveform - mean
    
    if config.norm_by_std:
        std = waveform.std(axis=-1, keepdims=True)
        std = np.where(std < config.eps, 1.0, std)
        waveform = waveform / std
    
    if config.norm_by_max:
        abs_max = np.abs(waveform).max()
        abs_max = np.where(abs_max < config.eps, 1.0, abs_max)
        waveform = waveform / abs_max
    
    if config.signed_sqrt:
        waveform = np.sign(waveform) * np.abs(waveform) ** config.signed_sqrt_factor
    
    return waveform


def apply_filter_config(waveform: np.ndarray, config: FilterConfig) -> np.ndarray:
    """
    Apply frequency filtering to waveform using config.
    
    Parameters
    ----------
    waveform : np.ndarray
        Waveform array of shape (channels, samples)
    config : FilterConfig
        Filter configuration
    
    Returns
    -------
    np.ndarray
        Filtered waveform
    """
    nyquist = config.sample_rate / 2.0
    
    if config.highpass_freq is not None and config.lowpass_freq is not None:
        low = config.highpass_freq / nyquist
        high = config.lowpass_freq / nyquist
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        if low < high:
            sos = signal.butter(config.filter_order, [low, high], btype='band', output='sos')
        else:
            return waveform
    elif config.highpass_freq is not None:
        freq = config.highpass_freq / nyquist
        freq = max(0.001, min(freq, 0.999))
        sos = signal.butter(config.filter_order, freq, btype='high', output='sos')
    elif config.lowpass_freq is not None:
        freq = config.lowpass_freq / nyquist
        freq = max(0.001, min(freq, 0.999))
        sos = signal.butter(config.filter_order, freq, btype='low', output='sos')
    else:
        return waveform
    
    filtered = np.zeros_like(waveform)
    for i in range(waveform.shape[0]):
        filtered[i] = signal.sosfiltfilt(sos, waveform[i])
    
    return filtered
