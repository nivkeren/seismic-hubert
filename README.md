# Seismic HuBERT

Self-supervised learning for seismic waveforms using the HuBERT architecture.

This project adapts [HuBERT](https://arxiv.org/abs/2106.07447) (Hidden-Unit BERT), originally designed for speech representation learning, to work with seismic data from the [STEAD dataset](https://github.com/smousavi05/STEAD). The goal is to create a foundation model for seismic signal processing that can be fine-tuned for various downstream tasks.

## Features

- **Self-supervised pretraining**: Learn representations from unlabeled seismic waveforms
- **Multi-channel support**: Process 3-component seismograms (E, N, Z)
- **STEAD integration**: Ready-to-use PyTorch dataset for the Stanford Earthquake Dataset
- **Flexible normalization**: Multiple normalization methods with amplitude preservation
- **Transfer learning**: Initialize from pretrained HuBERT models

## Documentation

- [Normalization Guide](docs/normalization.md) - Detailed explanation of normalization options

## Installation

This project uses [pixi](https://prefix.dev/docs/pixi/overview) for environment management:

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Create the environment and install dependencies
pixi install

# Activate the environment
pixi shell
```

## Dataset

Download the STEAD dataset from the [official repository](https://github.com/smousavi05/STEAD):

- `merge.hdf5`: ~85 GB of 3-component waveforms
- `merge.csv`: Metadata for all traces

Place both files in the `STEAD/` directory.

## Usage

### Training

```bash
pixi run train --hdf5_path STEAD/merge.hdf5 --csv_path STEAD/merge.csv
```

Or with more options:

```bash
python -m seismic_hubert.train \
    --hdf5_path STEAD/merge.hdf5 \
    --csv_path STEAD/merge.csv \
    --channel Z \
    --batch_size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --wandb
```

### Using the Dataset

```python
from seismic_hubert.data import STEADDataset, create_stead_dataloader

# Load dataset with normalization and filtering
dataset = STEADDataset(
    hdf5_path="STEAD/merge.hdf5",
    csv_path="STEAD/merge.csv",
    trace_category="earthquake_local",  # or "noise" or None for all
    channel="Z",  # or "all" for 3-channel
    norm_mode="zscore",  # "zscore", "robust_zscore", "peak", "quantile", "log", "none"
    highpass_freq=1.0,  # Remove low-frequency noise
    lowpass_freq=40.0,  # Remove high-frequency noise
    min_magnitude=3.0,
    max_distance_km=100,
)

# Get a sample
sample = dataset[0]
print(f"Waveform shape: {sample['waveform'].shape}")
print(f"Category: {sample['trace_category']}")

# Amplitude statistics (for magnitude/distance prediction)
print(f"Log max amplitude: {sample['amplitude_stats']['log_max_amp']:.2f}")
print(f"Log energy: {sample['amplitude_stats']['log_energy']:.2f}")

# Create a dataloader
dataloader = create_stead_dataloader(
    hdf5_path="STEAD/merge.hdf5",
    csv_path="STEAD/merge.csv",
    batch_size=32,
)
```

See [Normalization Guide](docs/normalization.md) for detailed explanation of options.

### Using the Model

```python
import torch
from seismic_hubert.models import SeismicHubert, SeismicHubertConfig

# Create model
config = SeismicHubertConfig(
    num_channels=1,  # Z channel only
    hidden_size=768,
    num_hidden_layers=12,
)
model = SeismicHubert(config)

# Forward pass
waveform = torch.randn(4, 1, 6000)  # batch=4, channels=1, samples=6000 (60s @ 100Hz)
outputs = model(waveform)
features = outputs["last_hidden_state"]
print(f"Feature shape: {features.shape}")
```

## Project Structure

```
seismic-hubert/
├── pixi.toml                    # Environment and dependencies
├── README.md
├── docs/
│   └── normalization.md        # Normalization guide
├── notebooks/
│   └── explore_stead.ipynb     # Dataset exploration
├── STEAD/                       # Dataset directory
│   ├── merge.hdf5              # Waveform data
│   └── merge.csv               # Metadata
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── stead_dataset.py    # STEAD PyTorch dataset
    │   ├── utils.py            # Normalization functions
    │   └── visualization.py    # Plotting utilities
    └── models/
        ├── __init__.py
        └── seismic_hubert.py   # HuBERT architecture
```

## Architecture

Seismic HuBERT adapts the original HuBERT architecture for seismic data:

1. **Feature Encoder**: CNN layers that downsample the 100 Hz waveform
2. **Transformer Encoder**: Standard transformer with masked self-attention
3. **Pretraining**: Masked prediction of cluster assignments (similar to MLM)

Key differences from audio HuBERT:
- Lower sample rate (100 Hz vs 16 kHz)
- Multi-channel input support (E, N, Z seismogram components)
- Adjusted CNN strides for seismic frequencies

## Downstream Tasks

After pretraining, the model can be fine-tuned for:
- **Phase picking**: P and S wave arrival time detection
- **Event detection**: Earthquake vs noise classification
- **Magnitude estimation**: Predict earthquake magnitude
- **Source characterization**: Depth, distance, mechanism

## References

- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
- [STEAD: A Global Data Set of Seismic Signals for AI](https://github.com/smousavi05/STEAD)
- [Earthquake Transformer](https://www.nature.com/articles/s41467-020-17591-w)

## License

MIT License
