# SeismicBenchDataset Implementation Notes

## Architecture

The `SeismicBenchDataset` provides a unified PyTorch interface for both local STEAD data and remote SeisBench benchmark datasets.

### How It Works

1. **STEAD Local Loading**
   - When `hdf5_path` and `csv_path` are provided, `SeismicBenchDataset` wraps `STEADDataset`
   - This avoids SSL certificate issues since no downloads are needed
   - Applies consistent preprocessing (normalization, filtering) to the data

2. **Remote Datasets**
   - For other datasets (ethz, lendb, geofon, etc.), uses native seisbench classes
   - Requires internet connection and valid SSL certificates
   - Data is cached locally for future use

### File Structure

```python
# Loading local STEAD
from data import SeismicBenchDataset

dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    channel='Z',
    norm_mode='zscore',
)
```

## Key Design Decisions

### 1. Wrapper for STEAD (Not Direct Integration)

**Why:** The seisbench STEAD class expects specific HDF5 structure and metadata columns that don't match our local STEAD format. The wrapper approach:
- ✓ Works with local STEAD files as-is
- ✓ No SSL issues
- ✓ Consistent with STEADDataset API
- ✓ Can be switched to direct seisbench loading if format is fixed in future

### 2. Unified Metadata Normalization

The `_normalize_metadata()` function converts various dataset formats to standard columns:
- `trace_name`, `trace_category`, `p_arrival_sample`, `s_arrival_sample`
- `source_magnitude`, `source_distance_km`, `source_depth_km`
- `station_code`, `network_code`, `split`, etc.

This allows the same preprocessing pipeline for all datasets.

### 3. Support for Extra Kwargs

```python
# Pass STEAD-specific parameters
dataset = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
    trace_category='earthquake_local',  # STEAD parameter
    max_samples=1000,
    subtract_mean=True,                 # STEAD parameter
)
```

Extra kwargs are passed to the underlying datasets via `**extra_kwargs`.

## Implementation Details

### `_load_seisbench_dataset()`

```python
def _load_seisbench_dataset(name, cache_root, hdf5_path, csv_path, **kwargs):
    if name == 'stead' and (hdf5_path or csv_path):
        # Wrap STEADDataset for local files
        return STEADDataset(hdf5_path=hdf5_path, csv_path=csv_path, **kwargs)
    else:
        # Use native seisbench class
        return sbd.<DatasetClass>(**kwargs)
```

### Waveform Retrieval

The `__getitem__` method handles both types of datasets:

```python
def __getitem__(self, idx):
    sb_dataset = self._sb_datasets[dataset_name]
    
    if hasattr(sb_dataset, 'get_waveforms'):
        # SeiBench datasets
        waveform = sb_dataset.get_waveforms(orig_idx)
    else:
        # STEADDataset (wrapped)
        sample = sb_dataset[orig_idx]
        waveform = sample['waveform'].numpy()
```

## Performance Considerations

- Metadata loading and normalization happens once during initialization
- Waveforms are loaded on-demand in `__getitem__`
- Filtering is applied to the combined metadata before creating the dataset
- Use `max_samples` and `max_samples_per_dataset` to limit dataset size for development

## Future Improvements

1. **Direct SeiBench STEAD Integration** - If seisbench updates its STEAD format
2. **Lazy Metadata Loading** - For very large combined datasets
3. **Parallel Dataset Loading** - When combining multiple remote datasets
4. **Caching Strategy** - Smart caching of frequently accessed samples

## Testing

Run comprehensive tests with:
```bash
cd /Users/nivk/Projects/seismic-hubert
pixi run python -c "
from src.data import SeismicBenchDataset
ds = SeismicBenchDataset('stead', hdf5_path='STEAD/merge.hdf5', csv_path='STEAD/merge.csv')
print(f'Loaded {len(ds)} samples')
sample = ds[0]
print(f'Sample shape: {sample[\"waveform\"].shape}')
"
```
