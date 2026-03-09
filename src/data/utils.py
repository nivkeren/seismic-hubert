from __future__ import annotations

import numpy as np
from typing import Optional
from scipy import signal


def normalize_waveform(
    waveform: np.ndarray,
    subtract_mean: bool = True,
    norm_by_std: bool = False,
    norm_by_max: bool = False,
    signed_sqrt: bool = False,
    signed_sqrt_factor: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply normalization to waveform.
    
    Normalization steps are applied in order:
    1. subtract_mean
    2. norm_by_std
    3. norm_by_max
    4. signed_sqrt
    
    Parameters
    ----------
    waveform : np.ndarray
        Waveform array of shape (channels, samples) or (samples,)
    subtract_mean : bool
        Subtract the mean from each channel
    norm_by_std : bool
        Divide by standard deviation
    norm_by_max : bool
        Divide by the absolute maximum value
    signed_sqrt : bool
        Apply signed power transform: sign(x) * |x|^factor
    signed_sqrt_factor : float
        Exponent for signed power transform (0.5 = sqrt)
    eps : float
        Small epsilon to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Normalized waveform (same shape as input)
    """
    if subtract_mean:
        mean = waveform.mean(axis=-1, keepdims=True)
        waveform = waveform - mean
    
    if norm_by_std:
        std = waveform.std(axis=-1, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        waveform = waveform / std
    
    if norm_by_max:
        abs_max = np.abs(waveform).max()
        abs_max = abs_max if abs_max > eps else 1.0
        waveform = waveform / abs_max
    
    if signed_sqrt:
        if signed_sqrt_factor > 1.0:
            signed_sqrt_factor = 1.0 / signed_sqrt_factor
        waveform = np.sign(waveform) * (np.abs(waveform)**signed_sqrt_factor)
    
    return waveform


def apply_filter(
    waveform: np.ndarray,
    highpass_freq: Optional[float] = None,
    lowpass_freq: Optional[float] = None,
    filter_order: int = 4,
    sample_rate: int = 100,
) -> np.ndarray:
    """
    Apply frequency filtering to waveform.
    
    Uses Butterworth filters from scipy.signal.
    
    Parameters
    ----------
    waveform : np.ndarray
        Waveform array of shape (channels, samples) or (samples,)
    highpass_freq : float | None
        High-pass filter cutoff frequency in Hz (removes frequencies below this)
    lowpass_freq : float | None
        Low-pass filter cutoff frequency in Hz (removes frequencies above this)
    filter_order : int
        Order of the Butterworth filter
    sample_rate : int
        Sample rate of the waveform in Hz
    
    Returns
    -------
    np.ndarray
        Filtered waveform (same shape as input)
    """
    if highpass_freq is None and lowpass_freq is None:
        return waveform
    
    # Handle 1D input
    is_1d = waveform.ndim == 1
    if is_1d:
        waveform = waveform[np.newaxis, :]
    
    nyquist = sample_rate / 2.0
    
    if highpass_freq is not None and lowpass_freq is not None:
        low = highpass_freq / nyquist
        high = lowpass_freq / nyquist
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        if low >= high:
            return waveform[0] if is_1d else waveform
        sos = signal.butter(filter_order, [low, high], btype='band', output='sos')
    elif highpass_freq is not None:
        freq = highpass_freq / nyquist
        freq = max(0.001, min(freq, 0.999))
        sos = signal.butter(filter_order, freq, btype='high', output='sos')
    else:  # lowpass_freq is not None
        freq = lowpass_freq / nyquist
        freq = max(0.001, min(freq, 0.999))
        sos = signal.butter(filter_order, freq, btype='low', output='sos')
    
    filtered = np.zeros_like(waveform)
    for i in range(waveform.shape[0]):
        filtered[i] = signal.sosfiltfilt(sos, waveform[i])
    
    return filtered[0] if is_1d else filtered


# =============================================================================
# Additional Normalization Methods
# =============================================================================

def log_compress(
    waveform: np.ndarray,
    subtract_mean: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Log compression: sign(x) * log(1 + scale * |x|)
    
    Compresses dynamic range while preserving sign and relative amplitudes.
    Good for data spanning many orders of magnitude.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform
    subtract_mean : bool
        Whether to subtract mean before compression
    scale : float
        Scaling factor before log (higher = more compression)
    
    Returns
    -------
    np.ndarray
        Log-compressed waveform
    """
    if subtract_mean:
        waveform = waveform - waveform.mean(axis=-1, keepdims=True)
    return np.sign(waveform) * np.log1p(scale * np.abs(waveform))


def quantile_normalize(
    waveform: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Quantile normalization: robust to outliers.
    
    Divides by the inter-quantile range instead of max or std.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform
    lower_quantile : float
        Lower quantile for range calculation
    upper_quantile : float
        Upper quantile for range calculation
    eps : float
        Small epsilon to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Quantile-normalized waveform
    """
    waveform = waveform - waveform.mean(axis=-1, keepdims=True)
    
    # Compute quantiles on absolute values
    abs_wf = np.abs(waveform)
    if waveform.ndim == 1:
        q_high = np.quantile(abs_wf, upper_quantile)
    else:
        q_high = np.quantile(abs_wf, upper_quantile, axis=-1, keepdims=True)
    
    q_high = np.maximum(q_high, eps)
    return waveform / q_high


def robust_zscore(
    waveform: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Robust z-score using median and MAD (median absolute deviation).
    
    More robust to outliers than standard z-score.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform
    eps : float
        Small epsilon to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Robust z-score normalized waveform
    """
    if waveform.ndim == 1:
        median = np.median(waveform)
        mad = np.median(np.abs(waveform - median))
    else:
        median = np.median(waveform, axis=-1, keepdims=True)
        mad = np.median(np.abs(waveform - median), axis=-1, keepdims=True)
    
    # Scale MAD to be consistent with std for normal distribution
    mad_std = 1.4826 * mad
    mad_std = np.maximum(mad_std, eps)
    
    return (waveform - median) / mad_std


def peak_normalize(
    waveform: np.ndarray,
    target_peak: float = 1.0,
    per_channel: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Peak normalization: scale so max absolute value equals target.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform of shape (channels, samples) or (samples,)
    target_peak : float
        Target peak value
    per_channel : bool
        If True, normalize each channel independently.
        If False, use global max across all channels.
    eps : float
        Small epsilon to avoid division by zero
    
    Returns
    -------
    np.ndarray
        Peak-normalized waveform
    """
    waveform = waveform - waveform.mean(axis=-1, keepdims=True)
    
    if per_channel and waveform.ndim > 1:
        # Per-channel: each channel scaled to [-1, 1]
        peak = np.abs(waveform).max(axis=-1, keepdims=True)
        peak = np.maximum(peak, eps)
    else:
        # Global: all channels scaled by same factor
        peak = np.abs(waveform).max()
        peak = max(peak, eps)
    
    return waveform * (target_peak / peak)


def mean_subtract(waveform: np.ndarray) -> np.ndarray:
    """
    Subtract mean only (no scaling).
    
    Centers the waveform around zero without changing amplitude.
    
    Parameters
    ----------
    waveform : np.ndarray
        Input waveform of shape (channels, samples) or (samples,)
    
    Returns
    -------
    np.ndarray
        Mean-subtracted waveform
    """
    return waveform - waveform.mean(axis=-1, keepdims=True)

