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

