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
- [Phase Picking Fine-Tuning](docs/phase_picking.md) - Methodology and running phase picking fine-tuning

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

Training is configured using [Hydra](https://hydra.cc/), which provides hierarchical YAML configs with command-line overrides.

#### Basic Usage

Commands work from both the project root and the `src/` directory:

```bash
# Using pixi (from project root)
pixi run train
pixi run train -- +experiment=overfit

# From project root
python src/train.py
python src/train.py +experiment=overfit

# From src/ directory
cd src
python train.py
python train.py +experiment=overfit

# Use an experiment preset
python train.py +experiment=overfit     # Quick test on small dataset
python train.py +experiment=advanced    # All domain-specific features
python train.py +experiment=debug       # Minimal config for debugging
```

#### Override Any Parameter

```bash
# Override single values
python train.py training.max_epochs=50 training.batch_size=32

# Override multiple values
python train.py data.channel=all clustering.feature_mode=combined

# Combine experiment preset with overrides
python train.py +experiment=advanced training.accelerator=cpu
```

#### Hyperparameter Sweeps

```bash
# Sweep over learning rates
python train.py --multirun training.lr=1e-4,5e-5,1e-5

# Grid search
python train.py --multirun training.lr=1e-4,5e-5 model.num_layers=6,12
```

#### Configuration Structure

```
conf/
├── config.yaml      # All settings in one flat file
└── experiment/
    ├── overfit.yaml  # Quick convergence test
    ├── advanced.yaml # Full features enabled
    └── debug.yaml    # Minimal for debugging
```

#### Key Configuration Options

| Config Path | Default | Description |
|------------|---------|-------------|
| `data.channel` | `Z` | Channels: `Z` (vertical) or `all` (E/N/Z) |
| `data.max_samples` | null | Limit samples (null = all) |
| `model.num_clusters` | 100 | K-means vocabulary size |
| `clustering.feature_mode` | `spectrogram` | `spectrogram`, `stalta`, `combined`, etc. |
| `masking.mask_length` | 5 | Fixed mask length (frames) |
| `masking.schedule` | `constant` | `constant`, `linear`, `step`, `cosine` |
| `masking.distance_adaptive` | false | Per-sample mask based on distance |
| `training.accelerator` | `auto` | `auto`, `cpu`, `gpu`, `mps` |
| `training.precision` | `32` | `32`, `16-mixed`, `bf16-mixed` |
| `logging.logger` | `tensorboard` | `tensorboard`, `mlflow`, or `wandb` |
| `logging.run_name` | null | Name for this run |

#### Output Directory

Hydra automatically creates timestamped output directories:

```
outputs/
└── 2024-01-15/
    └── 14-30-22/
        ├── .hydra/          # Hydra internals
        ├── config.yaml      # Resolved config
        ├── kmeans.pkl       # K-means model
        ├── checkpoints/     # Model checkpoints
        └── tensorboard/     # Logs
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
├── conf/                        # Hydra configuration
│   ├── config.yaml             # All settings (flat file)
│   └── experiment/             # Experiment presets
│       ├── overfit.yaml
│       ├── advanced.yaml
│       └── debug.yaml
├── docs/
│   └── normalization.md        # Normalization guide
├── notebooks/
│   └── explore_stead.ipynb     # Dataset exploration
├── STEAD/                       # Dataset directory
│   ├── merge.hdf5              # Waveform data
│   └── merge.csv               # Metadata
├── tests/                       # Unit tests
└── src/
    ├── train.py                # Training script (Hydra)
    ├── data/
    │   ├── __init__.py
    │   ├── stead_dataset.py    # STEAD PyTorch dataset
    │   ├── clustering.py       # K-means feature extraction
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

### Adaptations from Audio HuBERT

| Aspect | Audio HuBERT | Seismic HuBERT | Rationale |
|--------|--------------|----------------|-----------|
| Sample rate | 16,000 Hz | 100 Hz | Seismic signals are low-frequency (0.1-50 Hz) |
| Input channels | 1 (mono) | 1-3 (Z or E/N/Z) | 3-component seismograms capture wave polarization |
| CNN stride | 320x total | 32x total | Adjusted for lower sample rate |
| Typical input | 1-10 sec | 60 sec | Earthquakes unfold over longer timescales |
| Mask length | 10 frames | 10-20 frames | Longer spans for seismic event structure |

### Domain-Specific Features for Clustering

The K-means clustering that generates training targets supports multiple feature extraction modes optimized for seismic data:

| Feature Mode | Description | Best For |
|--------------|-------------|----------|
| `spectrogram` | STFT-based frequency content (default) | General purpose, similar to audio HuBERT |
| `stalta` | STA/LTA ratio for transient detection | Capturing P-wave onsets and impulsive signals |
| `frequency_bands` | Energy in seismic bands (0.1-1, 1-5, 5-20, 20-40 Hz) | Distinguishing teleseismic vs local events |
| `multi_channel` | Cross-channel polarization features | P vs S wave discrimination, back-azimuth |
| `combined` | All features concatenated | Richest representation for diverse signals |

**STA/LTA Features**: The Short-Term Average / Long-Term Average ratio is a classic seismological method for detecting transient energy changes. It naturally captures the impulsive onset of earthquake waves.

**Frequency Band Features**: Seismic events have characteristic frequency content based on source depth and distance:
- Teleseismic events (>1000 km): dominated by 0.1-1 Hz
- Regional events (100-1000 km): 1-5 Hz
- Local events (<100 km): 5-40 Hz

**Multi-Channel Polarization**: 3-component seismograms enable analysis of particle motion:
- P-waves: vertical motion dominant (high Z/horizontal ratio)
- S-waves: horizontal motion dominant (low Z/horizontal ratio)
- Rectilinearity: linear vs elliptical particle motion

### Masking Strategy & Curriculum Learning

Seismic signals have different temporal structure than speech:
- Speech phonemes: ~50-100 ms
- Seismic P-wave onset: ~100-500 ms
- S-wave arrival: ~1-2 seconds after P
- Full earthquake: ~10-60 seconds

At 32x downsampling from 100 Hz input, each frame represents **0.32 seconds**.

**Why curriculum learning for mask length?**
- A 3+ second mask could cover an entire P-wave arrival, making prediction impossible
- Short masks force the model to learn fine-grained waveform structure first
- Progressively longer masks teach event-level context as training advances

#### Automatic Mask Scheduling

Override `masking.schedule` to enable curriculum learning:

| Schedule | Behavior | Example (3→12 over 100 epochs) |
|----------|----------|--------------------------------|
| `constant` | Fixed mask length | Always 5 frames |
| `linear` | Gradual increase each epoch | Epoch 0: 3, Epoch 50: 7, Epoch 100: 12 |
| `step` | 3 discrete stages (33%/66%) | Epochs 0-32: 3, 33-65: 7, 66-100: 12 |
| `cosine` | Slow start/end, faster middle | Smooth S-curve progression |

```bash
# Enable linear mask scheduling
python train.py masking.schedule=linear masking.mask_length_start=3 masking.mask_length_end=12

# Or use the advanced preset which enables curriculum learning
python train.py +experiment=advanced
```

#### What the model learns at each stage:

| Mask Length | Duration | Learning Objective |
|-------------|----------|-------------------|
| 3 frames | ~1.0 sec | Local waveform patterns, partial P-wave onset |
| 6 frames | ~1.9 sec | P-wave structure, early S-wave hints |
| 9 frames | ~2.9 sec | P-to-S transitions, coda characteristics |
| 12 frames | ~3.8 sec | Full event structure, magnitude patterns |

The mask length is logged as `mask_length` metric, visible in TensorBoard/MLflow/W&B.

#### Distance-Adaptive Masking

Close earthquakes have shorter P-to-S intervals than distant ones:

| Distance | P-to-S Interval | Appropriate Mask |
|----------|-----------------|------------------|
| 10 km | ~1.3 sec | 2-4 frames |
| 50 km | ~6 sec | 6-10 frames |
| 100 km | ~12 sec | 10-15 frames |

With `masking.distance_adaptive=true`, each sample gets a mask length proportional to its source distance:

```bash
# Enable distance-adaptive masking
python train.py masking.distance_adaptive=true

# Or use the advanced preset which enables it
python train.py +experiment=advanced
```

**Combined with epoch scheduling:** When both `distance_adaptive` and a non-constant `schedule` are used, the maximum mask length scales with epoch progress. This means:
- Early epochs: close events get 2 frames, distant events get fewer than 15
- Late epochs: close events still get 2 frames, but distant events get the full 15

This prevents the model from seeing overly long masks for any event early in training.

## Downstream Tasks

After pretraining, the foundation model can be fine-tuned for various downstream tasks using the features it learned.

Currently supported implementations:
- **[Phase picking](docs/phase_picking.md)**: Detect P and S wave arrival times using a linear probe or a complex CNN head.

Other potential downstream tasks you could implement include:
- **Event detection**: Earthquake vs noise classification
- **Magnitude estimation**: Predict earthquake magnitude
- **Source characterization**: Depth, distance, mechanism

## References

- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
- [STEAD: A Global Data Set of Seismic Signals for AI](https://github.com/smousavi05/STEAD)
- [Earthquake Transformer](https://www.nature.com/articles/s41467-020-17591-w)

## License

MIT License
