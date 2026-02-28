# Malaria Detection using CNN

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A professional open-source toolkit for malaria detection using a Convolutional Neural Network (CNN). This project provides a complete pipeline to download NIH's Malaria Cell Image dataset, preprocess the images, and train a baseline CNN classifier, alongside model evaluation and interpretability tools (Grad-CAM).

## Project Structure

```
malaria-detection-cnn/
├── src/
│   ├── data.py         # Data handling and downloading
│   ├── model.py        # Baseline CNN architecture
│   ├── train.py        # Training pipeline
│   └── utils.py        # Plotting, evaluation and Grad-CAM tools
├── notebook.ipynb      # Guided tutorial and interactive pipeline
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore configuration
└── README.md           # This file
```

## Setup Instructions

### 1. Clone the repository and install dependencies
```bash
git clone https://github.com/your-username/malaria-detection-cnn.git
cd malaria-detection-cnn

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset
The code comes with an automated downloader. By running the training script or the notebook, the dataset will be downloaded and extracted directly from the NIH servers into `cell_images/`. 

## How to Run

### Command Line Interface

You can trigger the entire training pipeline, including downloading the dataset (if not locally present), splitting data, and training the model using the following command:

```bash
python -m src.train --epochs 15 --batch_size 32 --img_size 64
```

This will automatically:
1. Load images into training, validation, and testing generators.
2. Build the baseline CNN.
3. Train the model with early stopping.
4. Output the classification report and confusion matrix.
5. Save the trained model to `saved_models/malaria_final.keras`.

### Jupyter Notebook
For a guided, interactive walk-through of the pipeline including visual samples, training, evaluation, and interpretability maps (Grad-CAM), launch `notebook.ipynb`:

```bash
jupyter notebook notebook.ipynb
```

## Features
- **Clean Structure:** Modular OS-independent codebase.
- **Data Augmentation:** Increases model robustness to overfitting.
- **Model Interpretability:** Features Grad-CAM visualization mappings to identify what the CNN focused on to classify parasitized cells.
