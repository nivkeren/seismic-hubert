# Using SeismicBenchDataset with Local STEAD Data

Great news! **You can now use `SeismicBenchDataset` with local STEAD data** without any SSL certificate issues.

## Quick Start

```python
from data import SeismicBenchDataset

# Load STEAD data using SeismicBenchDataset
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    channel='Z',
    norm_mode='zscore',
)

# Get a sample
sample = dataset[0]
waveform = sample['waveform']
```

## Features Available

### 1. Filtering
```python
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    trace_category='earthquake_local',
    min_magnitude=3.0,
    max_distance_km=100,
    min_sampling_rate=99,
)
```

### 2. Normalization Modes
```python
for norm_mode in ['none', 'zscore', 'robust_zscore', 'peak', 'peak_per_ch', 'quantile', 'log', 'mean']:
    dataset = SeismicBenchDataset(
        datasets='stead',
        hdf5_path='STEAD/merge.hdf5',
        csv_path='STEAD/merge.csv',
        norm_mode=norm_mode,
    )
```

### 3. Signal Processing
```python
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    channel='Z',
    highpass_freq=1.0,
    lowpass_freq=20.0,
    filter_order=4,
    target_length=6000,
)
```

### 4. Channel Selection
```python
# Vertical channel only
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    channel='Z',
)

# All channels (E, N, Z)
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    channel='all',
)
```

## Supported STEAD Parameters

All standard STEAD dataset parameters are supported:

```python
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    # STEAD-specific options
    trace_category='earthquake_local',  # or 'noise', etc.
    max_samples=1000,
    subtract_mean=True,  # STEAD option
    norm_by_std=True,    # STEAD option
    # Plus all the SeismicBenchDataset options above
)
```

## Dataset Statistics

```python
stats = dataset.get_stats()
print(stats)
# Output:
# {
#     'total_samples': 50,
#     'datasets': ['stead'],
#     'samples_per_dataset': {'stead': 50},
#     'categories': {'earthquake_local': 39, 'noise': 11},
#     'magnitude_range': (0.08, 4.3),
#     'distance_range_km': (0.01, 149.9),
#     ...
# }
```

## Why This is Useful

1. **Unified Interface** - Same code works for both local STEAD and remote SeisBench datasets
2. **No SSL Issues** - Local STEAD data doesn't require downloads
3. **Consistent Preprocessing** - All datasets use the same normalization and filtering pipeline
4. **Easy Switching** - When SSL issues are fixed, switch to remote datasets with one line:
   ```python
   # Just change this line (remove hdf5_path and csv_path)
   dataset = SeismicBenchDataset(datasets='ethz')
   ```

## Complete Example

See `notebooks/explore_seisbench.ipynb` for full examples with:
- Loading and filtering data
- Normalization comparison
- Filter comparison  
- Earthquake vs Noise visualization
- Multiple sample comparisons

## What's New

The `SeismicBenchDataset` class has been enhanced to:
- Accept `hdf5_path` and `csv_path` parameters for local STEAD files
- Automatically detect if loading STEAD vs remote datasets
- Support all STEAD-specific parameters via `**extra_kwargs`
- Provide a unified interface for all dataset types

## Note on Remote SeisBench Datasets

If/when SSL issues are resolved, you can also use:
```python
dataset = SeismicBenchDataset(datasets='ethz')      # ETHZ
dataset = SeismicBenchDataset(datasets='lendb')     # LENDB
dataset = SeismicBenchDataset(datasets='geofon')    # GEOFON
# etc.
```

See [SEISBENCH_SSL_ISSUE.md](SEISBENCH_SSL_ISSUE.md) for troubleshooting.
