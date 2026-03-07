"""
Visualization utilities for seismic waveforms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


SAMPLE_RATE = 100  # Hz
WAVEFORM_LENGTH = 6000  # 60 seconds at 100 Hz


def plot_waveform(
    sample: dict,
    show_arrivals: bool = True,
    show_spectrogram: bool = False,
    figsize: tuple[float, float] = (12, 8),
    channels: str = "all",
    title: Optional[str] = None,
    savefig: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> plt.Figure:
    """
    Plot a seismic waveform sample with optional phase arrivals and spectrogram.
    
    Parameters
    ----------
    sample : dict
        Sample dictionary from STEADDataset containing 'waveform' tensor
        and metadata fields like 'trace_name', 'trace_category', 
        'p_arrival_sample', 's_arrival_sample', etc.
    show_arrivals : bool
        Show P and S wave arrival times
    show_spectrogram : bool
        Show spectrogram below waveforms
    figsize : tuple
        Figure size (width, height)
    channels : str
        Which channels to plot: 'all', 'Z', 'E', or 'N'
    title : str, optional
        Custom title (default: auto-generated from metadata)
    savefig : str, optional
        Path to save figure
    sample_rate : int
        Sample rate in Hz (default: 100)
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    
    Examples
    --------
    >>> from data.stead_dataset import STEADDataset
    >>> from data.visualization import plot_waveform
    >>> dataset = STEADDataset(hdf5_path, csv_path)
    >>> sample = dataset[0]
    >>> fig = plot_waveform(sample, show_arrivals=True)
    >>> plt.show()
    """
    waveform = sample["waveform"]
    
    if hasattr(waveform, 'numpy'):
        waveform = waveform.numpy()
    
    if waveform.ndim == 2 and waveform.shape[0] in (1, 3):
        waveform = waveform.T
    
    n_samples = waveform.shape[0]
    n_channels_data = waveform.shape[1] if waveform.ndim > 1 else 1
    
    if waveform.ndim == 1:
        waveform = waveform.reshape(-1, 1)
    
    channel_names = ['E', 'N', 'Z']
    channel_colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    if channels == 'Z':
        plot_channels = [min(2, n_channels_data - 1)]
    elif channels == 'E':
        plot_channels = [0]
    elif channels == 'N':
        plot_channels = [min(1, n_channels_data - 1)]
    else:
        plot_channels = list(range(n_channels_data))
    
    n_plot_channels = len(plot_channels)
    n_rows = n_plot_channels * 2 if show_spectrogram else n_plot_channels
    
    height_ratios = [2, 1] * n_plot_channels if show_spectrogram else [1] * n_plot_channels
    
    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize,
        gridspec_kw={'height_ratios': height_ratios}
    )
    
    if n_rows == 1:
        axes = [axes]
    
    time = np.arange(n_samples) / sample_rate
    
    p_arrival = sample.get("p_arrival_sample")
    s_arrival = sample.get("s_arrival_sample")
    trace_category = sample.get("trace_category", "")
    
    for i, ch_idx in enumerate(plot_channels):
        ax_idx = i * 2 if show_spectrogram else i
        ax = axes[ax_idx]
        
        data = waveform[:, ch_idx]
        color = channel_colors[ch_idx] if ch_idx < len(channel_colors) else 'black'
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f'Ch{ch_idx}'
        
        ax.plot(time, data, color=color, linewidth=0.5)
        ax.set_ylabel(f'{ch_name}\n(counts)', fontsize=10)
        ax.set_xlim(0, time[-1])
        ax.grid(True, alpha=0.3)
        
        if show_arrivals and trace_category == "earthquake_local":
            if p_arrival is not None:
                p_time = p_arrival / sample_rate
                ax.axvline(p_time, color='blue', linewidth=2, label='P-arrival')
            
            if s_arrival is not None:
                s_time = s_arrival / sample_rate
                ax.axvline(s_time, color='red', linewidth=2, label='S-arrival')
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        if show_spectrogram:
            ax_spec = axes[ax_idx + 1]
            ax_spec.specgram(
                data, Fs=sample_rate, 
                cmap='viridis', 
                noverlap=64,
                NFFT=128
            )
            ax_spec.set_ylabel('Freq (Hz)', fontsize=10)
            ax_spec.set_ylim(0, sample_rate / 2)
            
            if i < len(plot_channels) - 1:
                ax_spec.set_xticklabels([])
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    
    if title is None:
        trace_name = sample.get("trace_name", "Unknown")
        if trace_category == "earthquake_local":
            mag = sample.get("source_magnitude")
            dist = sample.get("source_distance_km")
            if mag is not None and dist is not None:
                title = f"{trace_name}\nM{mag:.1f} @ {dist:.1f} km"
            else:
                title = trace_name
        else:
            title = f"{trace_name} ({trace_category})"
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches='tight')
    
    return fig


def plot_waveform_batch(
    samples: list[dict],
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (4, 3),
    channel: str = "Z",
    show_arrivals: bool = True,
    savefig: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> plt.Figure:
    """
    Plot multiple waveform samples in a grid.
    
    Parameters
    ----------
    samples : list[dict]
        List of sample dictionaries from STEADDataset
    ncols : int
        Number of columns in the grid
    figsize_per_plot : tuple
        Size of each subplot
    channel : str
        Which channel to plot: 'E', 'N', or 'Z'
    show_arrivals : bool
        Show P and S wave arrivals
    savefig : str, optional
        Path to save figure
    sample_rate : int
        Sample rate in Hz (default: 100)
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    
    Examples
    --------
    >>> from data.stead_dataset import STEADDataset
    >>> from data.visualization import plot_waveform_batch
    >>> dataset = STEADDataset(hdf5_path, csv_path)
    >>> samples = [dataset[i] for i in range(6)]
    >>> fig = plot_waveform_batch(samples, ncols=3)
    >>> plt.show()
    """
    n_plots = len(samples)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False
    )
    
    ch_idx = {'E': 0, 'N': 1, 'Z': 2}.get(channel, 2)
    
    for i, sample in enumerate(samples):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        
        waveform = sample["waveform"]
        if hasattr(waveform, 'numpy'):
            waveform = waveform.numpy()
        
        if waveform.ndim == 2 and waveform.shape[0] in (1, 3):
            waveform = waveform.T
        
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)
        
        n_samples = waveform.shape[0]
        time = np.arange(n_samples) / sample_rate
        
        actual_ch_idx = min(ch_idx, waveform.shape[1] - 1)
        data = waveform[:, actual_ch_idx]
        
        ax.plot(time, data, 'k', linewidth=0.5)
        ax.set_xlim(0, time[-1])
        
        trace_category = sample.get("trace_category", "")
        if show_arrivals and trace_category == "earthquake_local":
            p_arrival = sample.get("p_arrival_sample")
            s_arrival = sample.get("s_arrival_sample")
            
            if p_arrival is not None:
                ax.axvline(p_arrival / sample_rate, color='blue', linewidth=1)
            if s_arrival is not None:
                ax.axvline(s_arrival / sample_rate, color='red', linewidth=1)
        
        mag = sample.get("source_magnitude")
        if trace_category == "earthquake_local" and mag is not None:
            ax.set_title(f"M{mag:.1f}", fontsize=9)
        else:
            ax.set_title(trace_category or "unknown", fontsize=9)
        
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    for i in range(n_plots, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'STEAD Waveforms ({channel} channel)', fontsize=12)
    plt.tight_layout()
    
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches='tight')
    
    return fig
