"""
QM9 dataset loader for SchNet pretraining.

The loader wraps torch_geometric.datasets.QM9 and supports selecting one
target property from the first 12 QM9 regression targets.
"""

import argparse
import os
import urllib.request
import zipfile
from typing import Callable, Optional

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


QM9_ZIP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
QM9_UNCHARACTERIZED_URL = "https://ndownloader.figshare.com/files/3195404"
QM9_REQUIRED_RAW_FILES = ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]


QM9_PROPERTIES = {
    "dipole": {"index": 0, "unit": "D", "name": "Dipole moment"},
    "alpha": {"index": 1, "unit": "a_0^3", "name": "Isotropic polarizability"},
    "homo": {"index": 2, "unit": "eV", "name": "Highest occupied molecular orbital"},
    "lumo": {"index": 3, "unit": "eV", "name": "Lowest unoccupied molecular orbital"},
    "gap": {"index": 4, "unit": "eV", "name": "HOMO-LUMO gap"},
    "r2": {"index": 5, "unit": "a_0^2", "name": "Electronic spatial extent"},
    "zpve": {"index": 6, "unit": "eV", "name": "Zero point vibrational energy"},
    "u0": {"index": 7, "unit": "eV", "name": "Internal energy at 0K"},
    "u298": {"index": 8, "unit": "eV", "name": "Internal energy at 298K"},
    "h298": {"index": 9, "unit": "eV", "name": "Enthalpy at 298K"},
    "g298": {"index": 10, "unit": "eV", "name": "Free energy at 298K"},
    "cv": {"index": 11, "unit": "cal/mol/K", "name": "Heat capacity at 298K"},
}


def _download_file(url: str, path: str) -> None:
    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percentage = downloaded * 100 / total_size
        downloaded_mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {percentage:5.1f}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)", end="")

    urllib.request.urlretrieve(url, path, reporthook)
    print()


def _prepare_local_raw_files(root: str) -> None:
    """Prepare QM9 raw files under the configured dataset root."""
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "qm9.zip")
    missing_before_extract = [
        filename
        for filename in ["gdb9.sdf", "gdb9.sdf.csv"]
        if not os.path.exists(os.path.join(raw_dir, filename))
    ]
    if missing_before_extract and os.path.exists(zip_path):
        print(f"Extracting local QM9 archive: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)


class QM9Dataset:
    """A thin wrapper around PyG QM9 that exposes one selected target."""

    def __init__(
        self,
        root: str = "./pretrain_models/data/QM9",
        property_name: str = "cv",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.property_name = property_name

        if property_name not in QM9_PROPERTIES:
            raise ValueError(
                f"Unknown property: {property_name}. "
                f"Available properties: {list(QM9_PROPERTIES.keys())}"
            )

        self.property_info = QM9_PROPERTIES[property_name]
        self.property_index = self.property_info["index"]

        print(
            f"Loading QM9 dataset for property: {self.property_info['name']} "
            f"({property_name})"
        )

        raw_dir = os.path.join(root, "raw")
        _prepare_local_raw_files(root)

        missing_files = [
            filename
            for filename in QM9_REQUIRED_RAW_FILES
            if not os.path.exists(os.path.join(raw_dir, filename))
        ]

        if missing_files:
            print(f"Dataset raw files are missing in {raw_dir}: {missing_files}")
            print("PyG will try to download/process them automatically.")
        else:
            print(f"Found local QM9 raw files at: {raw_dir}")

        try:
            self.dataset = QM9(
                root=root,
                transform=transform,
                pre_transform=pre_transform,
            )
        except Exception as exc:
            print("\n" + "=" * 60)
            print(f"Error loading/downloading QM9 dataset: {exc}")
            print("=" * 60)
            print("\nManual download instructions:")
            print(f"1. Create directory: {raw_dir}")
            print(f"2. Download: {QM9_ZIP_URL}")
            print(f"   Save as: {os.path.join(raw_dir, 'qm9.zip')}")
            print("3. Extract qm9.zip into the raw directory.")
            print(f"4. Download: {QM9_UNCHARACTERIZED_URL}")
            print(f"   Save as: {os.path.join(raw_dir, 'uncharacterized.txt')}")
            print("5. Ensure these files exist:")
            for filename in QM9_REQUIRED_RAW_FILES:
                print(f"   - {os.path.join(raw_dir, filename)}")
            print("=" * 60 + "\n")
            raise

        print(f"Dataset loaded: {len(self.dataset)} molecules")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data = self.dataset[idx]
        data.y = data.y.view(-1)[self.property_index].view(1)
        return data

    def get_statistics(self) -> dict:
        targets = torch.stack([self[idx].y for idx in range(len(self))]).view(-1)
        return {
            "mean": targets.mean().item(),
            "std": targets.std().item(),
            "min": targets.min().item(),
            "max": targets.max().item(),
            "count": len(targets),
        }


def get_qm9_dataloaders(
    root: str = "./pretrain_models/data/QM9",
    property_name: str = "cv",
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 0,
    shuffle: bool = True,
) -> tuple:
    """Create train/validation/test DataLoaders for QM9."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    dataset = QM9Dataset(root=root, property_name=property_name)

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size : train_size + val_size].tolist()
    test_indices = indices[train_size + val_size :].tolist()

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, dataset


def download_qm9_dataset(root: str = "./pretrain_models/data/QM9") -> bool:
    """Download and prepare the raw files expected by torch_geometric.datasets.QM9."""
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    _prepare_local_raw_files(root)

    zip_path = os.path.join(raw_dir, "qm9.zip")
    uncharacterized_path = os.path.join(raw_dir, "uncharacterized.txt")

    print("=" * 60)
    print("QM9 Dataset Downloader")
    print("=" * 60)
    print(f"Target directory: {raw_dir}\n")

    try:
        if os.path.exists(zip_path):
            print(f"qm9.zip already exists: {zip_path}")
        else:
            print(f"Downloading qm9.zip from {QM9_ZIP_URL}")
            _download_file(QM9_ZIP_URL, zip_path)

        print("Extracting qm9.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        if os.path.exists(uncharacterized_path):
            print(f"uncharacterized.txt already exists: {uncharacterized_path}")
        else:
            print(f"Downloading uncharacterized.txt from {QM9_UNCHARACTERIZED_URL}")
            _download_file(QM9_UNCHARACTERIZED_URL, uncharacterized_path)

        missing_files = [
            filename
            for filename in QM9_REQUIRED_RAW_FILES
            if not os.path.exists(os.path.join(raw_dir, filename))
        ]
        if missing_files:
            print(f"QM9 download is incomplete. Missing files: {missing_files}")
            return False

    except Exception as exc:
        print(f"Failed to download or prepare QM9: {exc}")
        print("\nPlease download manually:")
        print(f"  {QM9_ZIP_URL} -> {zip_path}")
        print(f"  {QM9_UNCHARACTERIZED_URL} -> {uncharacterized_path}")
        print(f"  unzip {zip_path} into {raw_dir}")
        return False

    print("\nQM9 dataset download completed.")
    print("Required raw files:")
    for filename in QM9_REQUIRED_RAW_FILES:
        print(f"  - {os.path.join(raw_dir, filename)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QM9 dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="./pretrain_models/data/QM9",
        help="Root directory for QM9 dataset",
    )
    args = parser.parse_args()

    success = download_qm9_dataset(root=args.root)
    if not success:
        print("\nDataset download failed. Please check the error messages above.")
        return

    print("\nVerifying dataset...")
    try:
        dataset = QM9Dataset(root=args.root, property_name="cv")
        stats = dataset.get_statistics()
        print("\nDataset verification successful.")
        print(f"  Total molecules: {stats['count']}")
        print("  Property: CV (Heat capacity at 298K)")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    except Exception as exc:
        print(f"\nDataset verification failed: {exc}")
        print("The dataset may be incomplete. Please check the files.")


if __name__ == "__main__":
    main()
