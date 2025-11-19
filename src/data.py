# src/data.py
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_train_val_test_split(
    src_dir,
    out_dir="data",
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
    copy=True
):
    """
    Splits dataset in src_dir (two subfolders expected, e.g. Parasitized/ Uninfected)
    into out_dir/train/<class>, out_dir/val/<class>, out_dir/test/<class>.
    If copy=False will move files (careful).
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
        def _transfer(lst, dst):
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
