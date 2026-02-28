# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
from typing import List, Tuple, Any, Optional, Union
import random
from matplotlib.image import imread

def visualize_samples(folder: Union[str, Path], n_samples: int = 6):
    """
    Visualizes random samples from a given directory.
    
    Args:
        folder: Path to the directory containing images.
        n_samples: Number of samples to visualize.
    """
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.is_file()]
    files = random.sample(files, min(n_samples, len(files)))
    
    plt.figure(figsize=(15, 3))
    for i, f in enumerate(files):
        img = imread(str(f))
        plt.subplot(1, min(n_samples, len(files)), i + 1)
        plt.imshow(img)
        plt.title(f.parent.name)
        plt.axis('off')
    plt.show()

def plot_history(history: Any):
    """
    Plots training and validation loss and accuracy from history.
    
    Args:
        history: Keras history object returned by model.fit()
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

def plot_confusion(cm: np.ndarray, classes: List[str], normalize: bool = False, title: str = 'Confusion matrix'):
    """
    Plots a confusion matrix.
    
    Args:
        cm: Confusion matrix array.
        classes: List of class names.
        normalize: Whether to normalize by row to show proportions.
        title: Title of the plot.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def evaluate_and_report(model: tf.keras.Model, generator: Any) -> Tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """
    Evaluates a model using a generator and computes metrics.
    
    Args:
        model: Trained Keras model.
        generator: Validation/test data generator.
        
    Returns:
        Tuple of (confusion_matrix, classification_report_str, y_true, y_pred).
    """
    preds = model.predict(generator, verbose=1)
    y_pred = (preds.ravel() > 0.5).astype(int)
    y_true = generator.classes  # requires flow_from_directory with shuffle=False
    
    cm = confusion_matrix(y_true, y_pred)
    class_names = list(generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    return cm, report, y_true, y_pred

def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: Optional[str] = None, pred_index: Optional[int] = None) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap for an image and model.
    This implementation safely executes layers manually to avoid Keras 3 Sequential graph bugs.
    """
    # Find last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
                
    if last_conv_layer_name is None:
        raise ValueError("No conv layer found; specify last_conv_layer_name")

    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(img_array)
        conv_outputs = None
        
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)
                
        predictions = x
        if pred_index is None:
            pred_index = 0
            
        pred = predictions[:, pred_index]

    if conv_outputs is None:
        raise ValueError(f"Layer {last_conv_layer_name} not found in model.")

    grads = tape.gradient(pred, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_path: Union[str, Path], heatmap: np.ndarray, alpha: float = 0.4, colormap: Any = plt.cm.jet) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto an image.
    
    Args:
        img_path: Path to the original image.
        heatmap: Grad-CAM heatmap generated by make_gradcam_heatmap.
        alpha: Opacity of the heatmap.
        colormap: Matplotlib colormap.
        
    Returns:
        Numpy array of the superimposed image.
    """
    img = image.load_img(str(img_path))
    img = image.img_to_array(img).astype(np.uint8)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap_img = image.array_to_img(np.expand_dims(heatmap, axis=-1)).resize((img.shape[1], img.shape[0]))
    heatmap_arr = image.img_to_array(heatmap_img)
    
    # apply colormap
    heatmap_colored = colormap(heatmap_arr[..., 0] / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    superimposed_img = heatmap_colored * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img
