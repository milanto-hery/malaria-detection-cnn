# src/model.py
from tensorflow.keras import layers, models, optimizers
from typing import Tuple

def build_baseline_cnn(input_shape: Tuple[int, int, int] = (64, 64, 3), dropout_rate: float = 0.5) -> models.Model:
    """
    Simple, clear baseline CNN for binary classification.
    
    Args:
        input_shape: Shape of the input images, defaults to (64, 64, 3).
        dropout_rate: Dropout rate for the fully connected layer.
        
    Returns:
        Compiled Keras Sequential model.
    """
    model = models.Sequential(name="baseline_cnn")
    
    # Block 1
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Fully Connected
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
