# src/data.py
import shutil
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Union, List

from sklearn.model_selection import train_test_split

def download_malaria_dataset(dest_dir: Union[str, Path] = "cell_images") -> Path:
    """
    Downloads and extracts the NIH Malaria dataset if not already present.
    
    Args:
        dest_dir: Directory to extract the 'cell_images' folder into.
        
    Returns:
        Path to the extracted dataset directory.
    """
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"Dataset already exists at {dest_dir}. Skipping download.")
        return dest_dir

    url = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
    
    zip_path = dest_dir.parent / "cell_images.zip"
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Failed to download from {url}: {e}")
        print("Please download manually from https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html and extract to 'cell_images'")
        return dest_dir
        
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir.parent)
    
    print("Download and extraction complete.")
    # Clean up zip file
    if zip_path.exists():
        zip_path.unlink()
        
    return dest_dir

def create_train_val_test_split(
    src_dir: Union[str, Path],
    out_dir: Union[str, Path] = "data",
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy: bool = True
) -> Tuple[Path, Path, Path]:
    """
    Splits dataset in src_dir (two subfolders expected, e.g. Parasitized/ Uninfected)
    into out_dir/train/<class>, out_dir/val/<class>, out_dir/test/<class>.
    If copy=False will move files (careful).
    
    Args:
        src_dir: Path to the source directory containing class subfolders.
        out_dir: Path to the output directory where splits will be created.
        val_ratio: Proportion of the dataset to include in the validation split.
        test_ratio: Proportion of the dataset to include in the test split.
        seed: Random seed for reproducibility.
        copy: Whether to copy files (True) or move them (False).
        
    Returns:
        Tuple containing Paths to the created train, val, and test directories.
    """
    random.seed(seed)
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    classes = [p.name for p in src_dir.iterdir() if p.is_dir()]
    if not classes:
        raise ValueError(f"No class subfolders found in {src_dir}")

    # create folders
    for split in ["train", "val", "test"]:
        for c in classes:
            (out_dir / split / c).mkdir(parents=True, exist_ok=True)

    for c in classes:
        files = list((src_dir / c).glob("*"))
        files = [f for f in files if f.is_file()]
        
        train_and_val, test_files = train_test_split(files, test_size=test_ratio, random_state=seed)
        train_files, val_files = train_test_split(train_and_val, test_size=val_ratio/(1-test_ratio), random_state=seed)
        
        def _transfer(lst: List[Path], dst: str):
            for src in lst:
                dest = out_dir / dst / c / src.name
                if copy:
                    shutil.copy(src, dest)
                else:
                    shutil.move(src, dest)
                    
        _transfer(train_files, "train")
        _transfer(val_files, "val")
        _transfer(test_files, "test")

    return out_dir / "train", out_dir / "val", out_dir / "test"
