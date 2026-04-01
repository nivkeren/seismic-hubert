# Branch Guide

## Branch Structure

This repository has two main branches:

### `main` (Default)
**Status:** Active development
**Focus:** STEADDataset - stable, working solution for local STEAD data

**What's included:**
- ✅ `STEADDataset` class - fully functional
- ✅ `notebooks/explore_stead.ipynb` - comprehensive exploration notebook
- ✅ All data utilities (normalization, filtering, visualization)
- ✅ No external dependencies for seisbench
- ✅ No SSL certificate issues

**Best for:** 
- Current development work
- Training models with STEAD data
- Data exploration and analysis

### `feature/seisbench-dataset`
**Status:** Experimental / Reference implementation
**Focus:** SeismicBenchDataset - unified interface for multiple datasets

**What's included:**
- 📚 `SeismicBenchDataset` class - wrapper for SeisBench + STEAD
- 📚 `notebooks/explore_seisbench.ipynb` - SeismicBenchDataset exploration
- 📚 Documentation:
  - `IMPLEMENTATION_NOTES.md` - architecture and design decisions
  - `SEISBENCH_SSL_ISSUE.md` - troubleshooting guide
  - `SEISBENCH_WITH_STEAD.md` - usage examples
- 📚 `setup_seisbench_cache.py` - cache setup utility
- 📚 `seisbench` as PyPI dependency
- ⚠️ SSL certificate requirement for remote datasets

**Best for:**
- Future work integrating multiple datasets
- Reference implementation
- Learning about seisbench architecture

## How to Use

### Stay on Main (Recommended)
```bash
# Already on main - just use STEADDataset
from data import STEADDataset

dataset = STEADDataset(
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
)
```

### Explore SeismicBenchDataset
```bash
# Switch to feature branch
git checkout feature/seisbench-dataset

# Now you can use:
from data import SeismicBenchDataset

# Local STEAD
ds_stead = SeismicBenchDataset(
    datasets='stead',
    hdf5_path='STEAD/merge.hdf5',
    csv_path='STEAD/merge.csv',
)

# Remote datasets (if SSL works)
ds_ethz = SeismicBenchDataset(datasets='ethz')
```

## Switching Branches

```bash
# See current branch
git branch

# Switch to feature branch
git checkout feature/seisbench-dataset

# Switch back to main
git checkout main

# List all branches
git branch -a
```

## Merging Future Work

When ready to integrate SeismicBenchDataset into main:

```bash
git checkout main
git merge feature/seisbench-dataset
```

## Key Differences

| Feature | main | feature/seisbench-dataset |
|---------|------|---------------------------|
| **STEADDataset** | ✅ Yes | ✅ Yes |
| **SeismicBenchDataset** | ❌ No | ✅ Yes |
| **Local STEAD** | ✅ Direct | ✅ Via wrapper |
| **Remote datasets** | ❌ No | ✅ Yes (SSL req) |
| **seisbench library** | ❌ No | ✅ Yes |
| **Complexity** | Low | Medium |
| **External deps** | Low | Higher |

## Why Two Branches?

1. **main** - Clean, focused, working solution for current work
2. **feature/seisbench-dataset** - Experimental, preserved for future reference
3. Easy to switch between approaches without losing work
4. Clear separation of concerns
5. Option to integrate later or explore alternatives

## Recommendations

- **For now:** Use `main` with `STEADDataset`
- **Revisit:** Check `feature/seisbench-dataset` branch when:
  - SSL certificate issues are resolved
  - Need to incorporate multiple SeisBench datasets
  - Want unified interface for heterogeneous datasets
