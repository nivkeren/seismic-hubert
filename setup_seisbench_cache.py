#!/usr/bin/env python3
"""
Setup script to symlink local STEAD files into seisbench cache.

This allows SeismicBenchDataset to load STEAD data without SSL issues.

Usage:
    python setup_seisbench_cache.py
"""

from pathlib import Path
import os
import sys

def setup_stead_cache(stead_dir: str = "STEAD"):
    """Setup seisbench cache with local STEAD files."""
    stead_path = Path(stead_dir).resolve()
    cache_dir = Path.home() / '.seisbench' / 'datasets' / 'stead'
    
    if not stead_path.exists():
        print(f"Error: STEAD directory not found at {stead_path}")
        return False
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    
    # Expected files
    required_files = {
        'merge.hdf5': 'waveforms.hdf5',
        'merge.csv': 'metadata.csv',
    }
    
    # Create symlinks
    for src_name, dst_name in required_files.items():
        src_file = stead_path / src_name
        dst_file = cache_dir / dst_name
        
        if not src_file.exists():
            print(f"Warning: {src_name} not found in {stead_path}")
            continue
        
        # Remove old symlink/file if exists
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
            print(f"  Removed old {dst_name}")
        
        # Create symlink
        os.symlink(src_file, dst_file)
        print(f"  ✓ Symlinked {dst_name} -> {src_file.name}")
    
    print(f"\nSetup complete! You can now load STEAD with:")
    print(f"  from data import SeismicBenchDataset")
    print(f"  dataset = SeismicBenchDataset(datasets='stead')")
    return True

if __name__ == '__main__':
    stead_dir = sys.argv[1] if len(sys.argv) > 1 else "STEAD"
    success = setup_stead_cache(stead_dir)
    sys.exit(0 if success else 1)
