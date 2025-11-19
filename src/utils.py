# src/utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pathlib

def visualize_samples(folder, n_samples=6):
    import random
    from matplotlib.image import imread
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
    files = random.sample(files, min(n_samples, len(files)))
    plt.figure(figsize=(15,3))
    for i,f in enumerate(files):
        img = imread(os.path.join(folder,f))
        plt.subplot(1, len(files), i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def plot_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.show()

def plot_confusion(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def evaluate_and_report(model, generator):
    # predict
    preds = model.predict(generator, verbose=1)
    y_pred = (preds.ravel() > 0.5).astype(int)
    y_true = generator.classes  # requires flow_from_directory with shuffle=False
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(generator.class_indices.keys()))
    return cm, report, y_true, y_pred

# Grad-CAM helper
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: preprocessed image array shape (1, H, W, C)
    model: tf.keras Model (expects model output is single sigmoid)
    last_conv_layer_name: optional string. If None, find last Conv2D layer.
    Returns heatmap (H, W) normalized 0..1
    """
    # find last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name and 'Conv2D' in layer.__class__.__name__ or 'conv2d' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("No conv layer found; specify last_conv_layer_name")

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        pred = predictions[:, pred_index]

    grads = tape.gradient(pred, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=plt.cm.jet):
    # load image
    img = image.load_img(img_path)
    img = image.img_to_array(img).astype(np.uint8)
    heatmap = np.uint8(255 * heatmap)
    heatmap = image.array_to_img(np.expand_dims(heatmap, axis=-1)).resize((img.shape[1], img.shape[0]))
    heatmap = image.img_to_array(heatmap)
    # apply colormap
    heatmap = colormap(heatmap[...,0]/255.0)[:,:,:3]
    heatmap = np.uint8(255*heatmap)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img
