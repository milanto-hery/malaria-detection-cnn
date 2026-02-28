# src/train.py
import argparse
import os
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Workaround for CUDA_ERROR_INVALID_HANDLE in some environments
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from src.model import build_baseline_cnn
from src.utils import plot_history, evaluate_and_report, plot_confusion

def create_generators(
    data_dir: Union[str, Path], 
    img_size: Tuple[int, int] = (64, 64), 
    batch_size: int = 32, 
    aug_params: Optional[Dict[str, Any]] = None, 
    seed: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Creates image data generators for training, validation, and testing.
    
    Args:
        data_dir: Path to the directory containing train/val/test subdirectories.
        img_size: Target image size for resizing.
        batch_size: Batch size for generators.
        aug_params: Dictionary of augmentation parameters for the training generator.
        seed: Random seed.
        
    Returns:
        Tuple of (train_gen, val_gen, test_gen).
    """
    data_dir = Path(data_dir)
    
    if aug_params is None:
        aug_params = dict(
            rotation_range=20, 
            zoom_range=0.05, 
            width_shift_range=0.05, 
            height_shift_range=0.05, 
            horizontal_flip=True
        )
        
    train_datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.0, **aug_params)
    val_datagen = ImageDataGenerator(rescale=1/255.0)
    test_datagen = ImageDataGenerator(rescale=1/255.0)

    train_gen = train_datagen.flow_from_directory(
        data_dir / 'train',
        target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir / 'val',
        target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        data_dir / 'test',
        target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )
    return train_gen, val_gen, test_gen


def main(args: argparse.Namespace):
    """
    Main training pipeline.
    """
    data_dir = Path(args.data_dir)
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    epochs = args.epochs

    if not data_dir.exists() or not list(data_dir.glob("*")):
        print(f"Data directory '{data_dir}' not found or empty.")
        from src.data import download_malaria_dataset, create_train_val_test_split
        print("Downloading and splitting dataset automatically...")
        dataset_dir = download_malaria_dataset("cell_images")
        create_train_val_test_split(
            src_dir=dataset_dir,
            out_dir=data_dir,
            val_ratio=0.2,
            test_ratio=0.1,
            copy=True
        )

    train_gen, val_gen, test_gen = create_generators(data_dir, img_size=img_size, batch_size=batch_size)

    model = build_baseline_cnn(input_shape=img_size + (3,), dropout_rate=0.5)
    model.summary()

    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tb_dir = Path("logs") / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    
    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(str(saved_models_dir / 'malaria_best.keras'), monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=str(tb_dir))
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    plot_history(history)

    # Evaluate on test
    cm, report, y_true, y_pred = evaluate_and_report(model, test_gen)
    print("Classification report:\n", report)
    plot_confusion(cm, classes=list(test_gen.class_indices.keys()), normalize=False)
    
    final_model_path = saved_models_dir / "malaria_final.keras"
    model.save(str(final_model_path))
    print(f"Saved {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline CNN for Malaria Detection")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder with train/val/test subfolders")
    parser.add_argument("--img_size", type=int, default=64, help="Target image size (dimension for width and height)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    args = parser.parse_args()
    main(args)
