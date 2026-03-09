# Waveform Normalization Guide

This document explains the normalization options available in the STEAD dataset and provides guidance on choosing the right method for your use case.

## Why Normalization Matters

Seismic waveform amplitudes vary by **orders of magnitude**:

| Earthquake | Amplitude Range |
|------------|-----------------|
| M1.0 (small) | ~0.001 - 0.1 counts |
| M4.0 (moderate) | ~10 - 1,000 counts |
| M7.0 (large) | ~10,000 - 1,000,000 counts |

Without normalization:
- Neural networks struggle with inputs spanning 10^6 range
- Large earthquakes dominate training
- Gradients become unstable

With normalization:
- All samples have similar scale
- Stable training
- But amplitude information may be lost

## Available Normalization Modes

```python
from data import STEADDataset

dataset = STEADDataset(
    hdf5_path="STEAD/merge.hdf5",
    csv_path="STEAD/merge.csv",
    norm_mode="zscore",  # Choose normalization method
)
```

### `zscore` (Default, Recommended)

**Z-score normalization**: Subtract mean, divide by standard deviation.

```
x_normalized = (x - mean(x)) / std(x)
```

| Property | Value |
|----------|-------|
| Output range | ~[-8, 8] (typically) |
| Mean | 0 |
| Std | 1 |
| Shape preserved | ✅ Yes (linear transform) |
| Amplitude info | ❌ Lost (saved in `amplitude_stats`) |

**Best for**: Self-supervised pretraining, classification tasks.

### `robust_zscore`

**Robust z-score**: Uses median and MAD (median absolute deviation) instead of mean/std.

```
x_normalized = (x - median(x)) / (1.4826 * MAD(x))
```

| Property | Value |
|----------|-------|
| Output range | ~[-20, 20] |
| Shape preserved | ✅ Yes |
| Robust to outliers | ✅ Yes |

**Best for**: Data with spikes or artifacts.

### `peak`

**Peak normalization**: Scale so global maximum absolute value equals 1.

```
x_normalized = (x - mean(x)) / max(|x|)  # global max across all channels
```

| Property | Value |
|----------|-------|
| Output range | [-1, 1] |
| Shape preserved | ✅ Yes |
| Channel relationship | ✅ Preserved (same scaling factor) |
| Sensitive to outliers | ⚠️ Yes (single spike affects all) |

**Best for**: When you need bounded range and want to preserve relative channel amplitudes.

### `peak_per_ch`

**Per-channel peak normalization**: Scale each channel independently to [-1, 1].

```
x_normalized[ch] = (x[ch] - mean(x[ch])) / max(|x[ch]|)  # per channel
```

| Property | Value |
|----------|-------|
| Output range | [-1, 1] per channel |
| Shape preserved | ✅ Yes |
| Channel relationship | ❌ Lost (each scaled independently) |

**Best for**: When each channel should be treated independently.

### `quantile`

**Quantile normalization**: Divide by 99th percentile instead of max.

```
x_normalized = (x - mean(x)) / quantile(|x|, 0.99)
```

| Property | Value |
|----------|-------|
| Output range | ~[-2, 2] |
| Shape preserved | ✅ Yes |
| Robust to outliers | ✅ Yes |

**Best for**: Data with occasional spikes.

### `log`

**Log compression**: Apply `sign(x) * log(1 + |x|)`.

```
x_compressed = sign(x) * log1p(|x|)
```

| Property | Value |
|----------|-------|
| Output range | ~[-10, 10] |
| Shape preserved | ❌ No (non-linear) |
| Amplitude info | ⚠️ Partially preserved |

**⚠️ Warning**: Log compression distorts waveform shape:
- A P-wave 100x larger than background becomes only ~3x larger after compression
- Relative amplitudes between phases are changed
- Correlation with original: ~0.7-0.8

**Best for**: Visualization only, not recommended for training.

### `mean`

**Mean subtraction only**: Subtract mean without scaling.

```
x_normalized = x - mean(x)
```

| Property | Value |
|----------|-------|
| Output range | Large (same as raw, but centered at 0) |
| Shape preserved | ✅ Yes |
| Amplitude info | ✅ Fully preserved |
| Training stability | ⚠️ May need careful learning rate tuning |

**Best for**: When you want to preserve absolute amplitudes but center the data.

### `none`

**No normalization**: Raw values.

| Property | Value |
|----------|-------|
| Output range | Very large (10^6) |
| Shape preserved | ✅ Yes |
| Training stability | ❌ Poor |

**Best for**: Testing, debugging, visualization.

## Preserving Amplitude Information

When using normalization (especially z-score), amplitude information is lost. To preserve it for downstream tasks (magnitude estimation, distance prediction), we automatically compute **amplitude statistics** before normalization:

```python
sample = dataset[0]

# Normalized waveform (shape preserved)
waveform = sample["waveform"]

# Pre-normalization amplitude statistics
stats = sample["amplitude_stats"]
print(stats)
# {
#     'log_max_amp': -2.3,   # log10(max|x|)
#     'log_std': -3.1,       # log10(std(x))
#     'log_energy': 2.4,     # log10(sum(x²))
#     'raw_max_amp': 0.005,  # actual max amplitude
# }
```

### Correlation with Magnitude

| Statistic | Correlation with Magnitude |
|-----------|---------------------------|
| `log_max_amp` | r ≈ 0.52 |
| `log_energy` | r ≈ 0.57 |

These statistics can be used by downstream models to predict magnitude/distance.

## Shape Preservation Comparison

| Method | Correlation with Raw | Shape Preserved |
|--------|---------------------|-----------------|
| zscore | 1.000 | ✅ Perfect |
| robust_zscore | ~0.99 | ✅ Yes |
| peak | 1.000 | ✅ Perfect |
| quantile | ~0.99 | ✅ Yes |
| log | ~0.72 | ❌ Distorted |

## Recommendations

### For Self-Supervised Pretraining

```python
dataset = STEADDataset(
    ...,
    norm_mode="zscore",
    highpass_freq=1.0,
    lowpass_freq=40.0,
)
```

- Model learns waveform **structure** (P/S waves, frequency content)
- Amplitude information available in `amplitude_stats` for fine-tuning

### For Magnitude Estimation

Use z-score normalized waveforms + amplitude_stats:

```python
class MagnitudeModel(nn.Module):
    def __init__(self, hubert_model):
        super().__init__()
        self.hubert = hubert_model
        self.amp_proj = nn.Linear(4, 768)  # Project amplitude stats
        self.head = nn.Linear(768, 1)
    
    def forward(self, waveform, amplitude_stats):
        features = self.hubert(waveform)["last_hidden_state"].mean(dim=1)
        amp_features = self.amp_proj(amplitude_stats)
        combined = features + amp_features
        return self.head(combined)
```

### For Phase Picking

```python
dataset = STEADDataset(
    ...,
    norm_mode="zscore",  # Shape matters most
)
```

Phase picking depends on waveform shape, not amplitude.

## Filtering

Filtering is applied **before** normalization and amplitude statistics computation:

```python
dataset = STEADDataset(
    ...,
    highpass_freq=1.0,   # Remove frequencies below 1 Hz
    lowpass_freq=40.0,   # Remove frequencies above 40 Hz
    filter_order=4,      # Butterworth filter order
)
```

### Recommended Filter Settings

| Application | High-pass | Low-pass |
|-------------|-----------|----------|
| General | 1.0 Hz | 40.0 Hz |
| Teleseismic | 0.01 Hz | 2.0 Hz |
| Local/Regional | 1.0 Hz | 20.0 Hz |
| Noise removal | 0.5 Hz | 45.0 Hz |

