# src/train.py
import argparse
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
import os
from src.model import build_baseline_cnn
from src.utils import plot_history, evaluate_and_report, plot_confusion
import datetime
import numpy as np

def create_generators(data_dir, img_size=(64,64), batch_size=32, aug_params=None, seed=42):
    if aug_params is None:
        aug_params = dict(rotation_range=20, zoom_range=0.05, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True)
    train_datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.0, **aug_params)
    val_datagen = ImageDataGenerator(rescale=1/255.0)
    test_datagen = ImageDataGenerator(rescale=1/255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )
    return train_gen, val_gen, test_gen

def main(args):
    data_dir = args.data_dir
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    epochs = args.epochs

    train_gen, val_gen, test_gen = create_generators(data_dir, img_size=img_size, batch_size=batch_size)

    model = build_baseline_cnn(input_shape=img_size+(3,), dropout_rate=0.5)
    model.summary()

    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tb_dir = Path("logs")/run_name
    tb_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('malaria_best.h5', monitor='val_loss', save_best_only=True),
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

    model.save("malaria_final.h5")
    print("Saved malaria_final.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder with train/val/test subfolders")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args)
