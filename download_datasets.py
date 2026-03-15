#!/usr/bin/env python3
"""
Download seismic datasets using SeisBench.

All datasets from: https://seisbench.readthedocs.io/en/stable/pages/data/benchmark_datasets.html

Usage:
    python download_datasets.py                     # List available datasets
    python download_datasets.py stead              # Download STEAD
    python download_datasets.py ethz instance      # Download multiple datasets
    python download_datasets.py --all              # Download all auto-downloadable datasets
    python download_datasets.py --list             # List datasets with details
    python download_datasets.py --category local   # Download datasets by category
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Dataset registry with metadata from SeisBench documentation
DATASETS = {
    # ===== Aftershock / Regional Datasets =====
    "aq2009_counts": {
        "class": "AQ2009Counts",
        "description": "2009 L'Aquila earthquake aftershocks (digital units)",
        "size": "~123GB",
        "samples": "~1.26M",
        "category": "regional",
        "manual_download": False,
    },
    "aq2009_gm": {
        "class": "AQ2009GM",
        "description": "2009 L'Aquila earthquake aftershocks (ground motion)",
        "size": "~123GB",
        "samples": "~1.26M",
        "category": "regional",
        "manual_download": False,
    },
    "bohemia": {
        "class": "BohemiaSaxony",
        "description": "NW Bohemia/Vogtland earthquake swarms (2009-2024)",
        "size": "~10GB",
        "samples": "~25K",
        "category": "regional",
        "manual_download": True,
        "manual_instructions": (
            "Requires EIDA token for WEBNET restricted data.\n"
            "See: https://seisbench.readthedocs.io/en/stable/pages/data/benchmark_datasets.html#bohemia"
        ),
    },
    
    # ===== California Datasets =====
    "ceed": {
        "class": "CEED",
        "description": "California Earthquake Event Dataset (1987-2023)",
        "size": "~575GB",
        "samples": "~4.1M",
        "category": "california",
        "manual_download": False,
        "warning": "Very large dataset (~575GB)",
    },
    "scedc": {
        "class": "SCEDC",
        "description": "Southern California Earthquake Data Center (2000-2020)",
        "size": "~660GB",
        "samples": "~8.1M",
        "category": "california",
        "manual_download": False,
        "warning": "Very large dataset (~660GB)",
    },
    
    # ===== Global / Teleseismic Datasets =====
    "crew": {
        "class": "CREW",
        "description": "Curated Regional Earthquake Waveforms (global coverage)",
        "size": "~1.1TB",
        "samples": "~1.6M",
        "category": "global",
        "manual_download": False,
        "warning": "Extremely large dataset (~1.1TB)",
    },
    "geofon": {
        "class": "GEOFON",
        "description": "GFZ Potsdam global monitoring (2009-2013)",
        "size": "~26GB",
        "samples": "~275K",
        "category": "global",
        "manual_download": False,
    },
    "mlaapde": {
        "class": "MLAAPDE",
        "description": "USGS global earthquake source parameters",
        "size": "~50GB",
        "samples": "~1.9M",
        "category": "global",
        "manual_download": False,
    },
    "neic": {
        "class": "NEIC",
        "description": "National Earthquake Information Center (superseded by MLAAPDE)",
        "size": "~12GB",
        "samples": "~1.3M",
        "category": "global",
        "manual_download": False,
        "deprecated": True,
    },
    "isc_ehb_depth": {
        "class": "ISC_EHB_DepthPhases",
        "description": "ISC-EHB depth phases (pP, sP, pwP)",
        "size": "~15GB",
        "samples": "~174K",
        "category": "global",
        "manual_download": False,
    },
    
    # ===== European Datasets =====
    "ethz": {
        "class": "ETHZ",
        "description": "Swiss Seismological Service (2013-2020)",
        "size": "~22GB",
        "samples": "~37K",
        "category": "europe",
        "manual_download": False,
    },
    
    # ===== Italian Datasets =====
    "instance_counts": {
        "class": "InstanceCounts",
        "description": "INGV Italian dataset (digital counts)",
        "size": "~160GB",
        "samples": "~1.2M",
        "category": "italy",
        "manual_download": False,
    },
    "instance_gm": {
        "class": "InstanceGM",
        "description": "INGV Italian dataset (ground motion units)",
        "size": "~310GB",
        "samples": "~1.2M",
        "category": "italy",
        "manual_download": False,
    },
    "instance_noise": {
        "class": "InstanceNoise",
        "description": "INGV Italian noise examples",
        "size": "~20GB",
        "samples": "~130K",
        "category": "italy",
        "manual_download": False,
    },
    "instance_combined": {
        "class": "InstanceCountsCombined",
        "description": "INGV Italian dataset (counts + noise combined)",
        "size": "~180GB",
        "samples": "~1.3M",
        "category": "italy",
        "manual_download": False,
    },
    
    # ===== Taiwan Dataset =====
    "cwa": {
        "class": "CWA",
        "description": "Central Weather Bureau Taiwan (CWASN + TSMIP)",
        "size": "~494GB",
        "samples": "~500K",
        "category": "asia",
        "manual_download": False,
        "warning": "Very large dataset (~494GB)",
    },
    
    # ===== South American Datasets =====
    "iquique": {
        "class": "Iquique",
        "description": "2014 Iquique Mw8.1 aftershocks (Chile)",
        "size": "~5GB",
        "samples": "~13K",
        "category": "south_america",
        "manual_download": False,
    },
    
    # ===== North American Datasets =====
    "lendb": {
        "class": "LENDB",
        "description": "Local Earthquake Network Database (global stations)",
        "size": "~20GB",
        "samples": "~1.25M",
        "category": "global",
        "manual_download": False,
    },
    "pnw": {
        "class": "PNW",
        "description": "Pacific Northwest velocity channels",
        "size": "~30GB",
        "samples": "~500K",
        "category": "north_america",
        "manual_download": False,
    },
    "pnw_accelerometers": {
        "class": "PNWAccelerometers",
        "description": "Pacific Northwest accelerometer data",
        "size": "~15GB",
        "samples": "~200K",
        "category": "north_america",
        "manual_download": False,
    },
    "pnw_noise": {
        "class": "PNWNoise",
        "description": "Pacific Northwest noise waveforms",
        "size": "~10GB",
        "samples": "~100K",
        "category": "north_america",
        "manual_download": False,
    },
    "pnw_exotic": {
        "class": "PNWExotic",
        "description": "Pacific Northwest exotic events (surface, thunder, sonic boom)",
        "size": "~5GB",
        "samples": "~50K",
        "category": "north_america",
        "manual_download": False,
    },
    "txed": {
        "class": "TXED",
        "description": "Texas Earthquake Dataset",
        "size": "~70GB",
        "samples": "~500K",
        "category": "north_america",
        "manual_download": False,
    },
    
    # ===== LFE Stack Datasets =====
    "lfe_cascadia": {
        "class": "LFEStacksCascadiaBostock2015",
        "description": "Low-frequency earthquake stacks - Cascadia",
        "size": "~500MB",
        "samples": "~2K",
        "category": "lfe",
        "manual_download": False,
    },
    "lfe_mexico": {
        "class": "LFEStacksMexicoFrank2014",
        "description": "Low-frequency earthquake stacks - Guerrero, Mexico",
        "size": "~2GB",
        "samples": "~11K",
        "category": "lfe",
        "manual_download": False,
    },
    "lfe_san_andreas": {
        "class": "LFEStacksSanAndreasShelly2017",
        "description": "Low-frequency earthquake stacks - San Andreas Fault",
        "size": "~500MB",
        "samples": "~2K",
        "category": "lfe",
        "manual_download": False,
    },
    
    # ===== Ocean Bottom Seismometer Datasets =====
    "obs": {
        "class": "OBS",
        "description": "Ocean-bottom seismometer benchmark (15 deployments)",
        "size": "~15GB",
        "samples": "~110K",
        "category": "obs",
        "manual_download": False,
    },
    "obst2024": {
        "class": "OBST2024",
        "description": "OBS dataset for OBSTransformer (11 deployments)",
        "size": "~10GB",
        "samples": "~60K",
        "category": "obs",
        "manual_download": False,
    },
    
    # ===== Induced Seismicity =====
    "pisdl": {
        "class": "PiSDL",
        "description": "Induced seismicity (hydraulic fracturing, geothermal, mining)",
        "size": "~35GB",
        "samples": "~142K",
        "category": "induced",
        "manual_download": False,
    },
    
    # ===== Volcanic Datasets =====
    "vcseis": {
        "class": "VCSEIS",
        "description": "Volcanic earthquakes (Alaska, Hawaii, N. California, Cascades)",
        "size": "~47GB",
        "samples": "~160K",
        "category": "volcanic",
        "manual_download": False,
    },
    
    # ===== STEAD (requires manual download) =====
    "stead": {
        "class": "STEAD",
        "description": "Stanford Earthquake Dataset - Global seismic signals",
        "size": "~70GB",
        "samples": "~1.2M",
        "category": "global",
        "manual_download": True,
        "manual_instructions": (
            "STEAD requires manual download from https://github.com/smousavi05/STEAD\n"
            "Download merge.csv and merge.hdf5, then provide the path via --basepath"
        ),
    },
}

# Category descriptions
CATEGORIES = {
    "global": "Global/Teleseismic datasets",
    "california": "California regional datasets",
    "europe": "European datasets",
    "italy": "Italian datasets (INSTANCE)",
    "asia": "Asian datasets",
    "north_america": "North American datasets",
    "south_america": "South American datasets",
    "regional": "Regional aftershock sequences",
    "lfe": "Low-frequency earthquake stacks",
    "obs": "Ocean-bottom seismometer datasets",
    "induced": "Induced seismicity datasets",
    "volcanic": "Volcanic earthquake datasets",
}


def get_dataset_class(name: str):
    """Dynamically import and return the dataset class."""
    import seisbench.data as sbd
    
    class_name = DATASETS[name]["class"]
    if hasattr(sbd, class_name):
        return getattr(sbd, class_name)
    else:
        raise ValueError(f"Dataset class {class_name} not found in seisbench.data")


def list_datasets(verbose: bool = False, category: Optional[str] = None) -> None:
    """List all available datasets."""
    print("\n" + "=" * 80)
    print("Available Seismic Datasets (SeisBench)")
    print("=" * 80)
    
    # Group by category
    by_category = {}
    for name, info in DATASETS.items():
        cat = info.get("category", "other")
        if category and cat != category:
            continue
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((name, info))
    
    for cat, datasets in sorted(by_category.items()):
        cat_desc = CATEGORIES.get(cat, cat.title())
        print(f"\n📁 {cat_desc}")
        print("-" * 40)
        
        for name, info in sorted(datasets):
            if info.get("manual_download"):
                status = "📋 manual"
            elif info.get("deprecated"):
                status = "⚠️  deprecated"
            else:
                status = "📥 auto"
            
            print(f"  {name:20} [{status:12}] {info['size']:>10} | {info['samples']:>8}")
            
            if verbose:
                print(f"  {'':20} {info['description']}")
                if info.get("warning"):
                    print(f"  {'':20} ⚠️  {info['warning']}")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(DATASETS)} datasets")
    print("\nUsage:")
    print("  python download_datasets.py <dataset> [<dataset> ...]")
    print("  python download_datasets.py --list                    # Detailed list")
    print("  python download_datasets.py --category volcanic       # By category")
    print("  python download_datasets.py --all                     # All < 100GB")
    print("  python download_datasets.py --all --include-large     # Include large datasets")
    print("\nCategories:", ", ".join(CATEGORIES.keys()))
    print()


def download_dataset(
    name: str,
    cache_dir: Path,
    basepath: Optional[Path] = None,
    force: bool = False,
) -> bool:
    """Download a single dataset."""
    
    if name not in DATASETS:
        print(f"❌ Unknown dataset: {name}")
        similar = [d for d in DATASETS.keys() if name in d or d in name]
        if similar:
            print(f"   Did you mean: {', '.join(similar)}?")
        return False
    
    info = DATASETS[name]
    dataset_cache = cache_dir / name
    
    print(f"\n{'='*70}")
    print(f"📦 Dataset: {name.upper()}")
    print(f"   {info['description']}")
    print(f"   Size: {info['size']}, Samples: {info['samples']}")
    print(f"   Category: {info.get('category', 'unknown')}")
    print(f"   Cache: {dataset_cache}")
    print("=" * 70)
    
    if info.get("deprecated"):
        print(f"⚠️  This dataset is deprecated. Consider using an alternative.")
    
    if info.get("warning"):
        print(f"⚠️  Warning: {info['warning']}")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipped.")
            return False
    
    # Check if already downloaded
    if dataset_cache.exists() and not force:
        print(f"ℹ️  Dataset already exists at {dataset_cache}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipped.")
            return True
    
    try:
        dataset_cache.mkdir(parents=True, exist_ok=True)
        DatasetClass = get_dataset_class(name)
        
        # Handle manual download datasets
        if info.get("manual_download"):
            if basepath is None:
                print(f"\n❌ {name.upper()} requires manual download.")
                print(info["manual_instructions"])
                print("\nProvide the path with: --basepath /path/to/downloaded/files")
                return False
            
            print(f"Converting from {basepath}...")
            data = DatasetClass(
                cache_root=str(cache_dir),
                download_kwargs={"basepath": str(basepath)},
            )
        else:
            print("Downloading... (this may take a while)")
            data = DatasetClass(cache_root=str(cache_dir))
        
        print(f"✅ Download complete!")
        print(f"   Path: {data.path}")
        print(f"   Samples: {len(data)}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_size(size_str: str) -> float:
    """Parse size string like '~70GB' to float in GB."""
    size_str = size_str.replace("~", "").strip().upper()
    if "TB" in size_str:
        return float(size_str.replace("TB", "")) * 1024
    elif "GB" in size_str:
        return float(size_str.replace("GB", ""))
    elif "MB" in size_str:
        return float(size_str.replace("MB", "")) / 1024
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download seismic datasets using SeisBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to download (e.g., stead ethz instance_counts)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets with details",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all auto-downloadable datasets (< 100GB by default)",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large datasets (> 100GB) when using --all",
    )
    parser.add_argument(
        "--category", "-c",
        choices=list(CATEGORIES.keys()),
        help="Filter or download datasets by category",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/seisbench"),
        help="Cache directory for datasets (default: data/seisbench)",
    )
    parser.add_argument(
        "--basepath",
        type=Path,
        help="Path to manually downloaded files (for STEAD, Bohemia)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset exists",
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_datasets(verbose=True, category=args.category)
        return 0
    
    # Handle no arguments
    if not args.datasets and not args.all and not args.category:
        list_datasets(verbose=False)
        return 0
    
    # Resolve cache directory
    cache_dir = args.cache_dir.resolve()
    print(f"\n📁 Cache directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to download
    if args.all or (args.category and not args.datasets):
        max_size = 1024 if args.include_large else 100  # GB
        datasets = []
        for name, info in DATASETS.items():
            # Skip manual download datasets
            if info.get("manual_download"):
                continue
            # Skip deprecated datasets
            if info.get("deprecated"):
                continue
            # Filter by category if specified
            if args.category and info.get("category") != args.category:
                continue
            # Filter by size
            size_gb = parse_size(info["size"])
            if size_gb > max_size:
                print(f"Skipping {name} ({info['size']}) - too large")
                continue
            datasets.append(name)
        
        if not args.include_large:
            print(f"Downloading datasets < 100GB (use --include-large for larger)")
    else:
        datasets = [d.lower().replace("-", "_") for d in args.datasets]
    
    if not datasets:
        print("No datasets to download.")
        return 0
    
    print(f"\nDatasets to download: {', '.join(datasets)}")
    
    # Download each dataset
    results = {}
    for name in datasets:
        success = download_dataset(
            name=name,
            cache_dir=cache_dir,
            basepath=args.basepath,
            force=args.force,
        )
        results[name] = success
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Summary:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\n   {successful}/{total} datasets downloaded successfully")
    print("=" * 70)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
