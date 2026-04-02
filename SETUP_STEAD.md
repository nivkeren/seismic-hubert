# STEAD Data Setup

The STEAD dataset files are NOT included in this repository because they are too large for Git. You need to download both files before training:

- ❌ `STEAD/merge.csv` - Metadata (~350MB)
- ❌ `STEAD/merge.hdf5` - Waveforms (~70GB)

## Get the Data Files

### Option 1: Manual Download (Recommended)

1. Go to: https://github.com/smousavi05/STEAD
2. Follow the download instructions to get both `merge.csv` and `merge.hdf5`
3. Place them in your project's STEAD directory:
   ```bash
   mkdir -p /Users/nivk/Projects/seismic-hubert/STEAD
   cp /path/to/merge.hdf5 /Users/nivk/Projects/seismic-hubert/STEAD/
   cp /path/to/merge.csv /Users/nivk/Projects/seismic-hubert/STEAD/
   ```

### Option 2: Download via Script

If you're on the `feature/seisbench-dataset` branch, you can try using the download script:

```bash
# Note: This requires seisbench library and internet connection
# May fail due to SSL certificates on some networks
python download_datasets.py stead --basepath /path/to/stead/files
```

## Verify Installation

Check that both files exist:

```bash
ls -lh STEAD/
# Should show:
# -rw-r--r-- merge.csv      (~350MB)
# -rw-r--r-- merge.hdf5     (~70GB)
```

## Run Training

Once both files are present:

```bash
# Activate environment
pixi shell

# Run training with overfit experiment
pixi run train -- +experiment=overfit

# Or other experiments
pixi run train -- +experiment=pretrain
```

## Troubleshooting

If you get "FileNotFoundError: merge.csv" or "merge.hdf5":
1. Confirm both files are in the `STEAD/` folder
2. Check file permissions: `ls -la STEAD/`
3. Verify they're not corrupted: `file STEAD/merge.hdf5`
