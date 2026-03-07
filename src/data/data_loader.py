from pathlib import Path
from torch.utils.data import DataLoader
from data.stead_dataset import STEADDataset, STEADCollator

def create_stead_dataloader(
    hdf5_path: str | Path,
    csv_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for the STEAD dataset.
    
    Parameters
    ----------
    hdf5_path : str | Path
        Path to STEAD HDF5 file
    csv_path : str | Path
        Path to STEAD CSV file
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers
    shuffle : bool
        Whether to shuffle the data
    **dataset_kwargs
        Additional arguments passed to STEADDataset
    
    Returns
    -------
    DataLoader
        PyTorch DataLoader for STEAD
    """
    dataset = STEADDataset(hdf5_path, csv_path, **dataset_kwargs)
    collator = STEADCollator(return_labels=True)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
