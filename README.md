# CNN on CIFAR-10 with Hyperparameter Ablation Study

This repository presents an end-to-end implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The project includes training, evaluation, feature map visualization, and a systematic ablation study to analyze the effect of key hyperparameters.

## Overview

The objective of this project is to build a configurable CNN model and study how different architectural and training choices affect performance. The implementation is done in PyTorch and supports GPU acceleration.

## Dataset

* Dataset: CIFAR-10
* Source: Hugging Face (`datasets` library)
* Total images: 60,000
* Image size: 32x32 (RGB)
* Classes: 10

The dataset is loaded directly using the Hugging Face API and wrapped into a PyTorch-compatible dataset.

## Data Preprocessing

* Random horizontal flip
* Random cropping with padding
* Normalization using CIFAR-10 mean and standard deviation

## Model Architecture

A flexible CNN architecture is implemented with the following components:

* Multiple convolutional layers
* Batch normalization
* ReLU activation
* Periodic max pooling
* Adaptive average pooling
* Fully connected classifier with dropout

The architecture is configurable in terms of:

* Number of convolutional layers
* Number of filters

## Training Setup

* Loss function: CrossEntropyLoss
* Optimizer: SGD with momentum
* Learning rate scheduler: StepLR
* GPU support (including multi-GPU with DataParallel)
* Reproducibility using fixed random seeds

Training tracks:

* Loss per epoch
* Training error per epoch

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

## Feature Map Visualization

Feature maps are extracted from intermediate convolutional layers to analyze learned representations:

* Early layers capture edges and color patterns
* Middle layers capture textures
* Deeper layers capture high-level features

## Ablation Study

A one-factor-at-a-time ablation study is conducted to analyze the impact of key hyperparameters.

### Hyperparameters Tested

* Learning Rate: 0.001, 0.01, 0.1
* Batch Size: 16, 32, 64
* Number of Filters: 16, 32, 64
* Number of Layers: 3, 5, 7

Each configuration is trained and evaluated independently.

## Model Comparison

The project compares:

* Baseline model (default parameters)
* Best single ablation configuration
* Retrained model using optimal hyperparameters

Performance is compared using:

* Accuracy
* Precision
* Recall
* F1-score

## Results

The implementation produces:

* Training loss and error curves
* Confusion matrices
* Tabular comparison of model performance
* Feature map visualizations

## Installation

Install dependencies:

pip install torch torchvision datasets scikit-learn matplotlib pandas tqdm

## Usage

Run the script:

python cnn_cifar10_ablation.py

The script will:

* Train the baseline model
* Evaluate performance
* Visualize feature maps
* Perform ablation study
* Retrain using best hyperparameters
* Output final comparison results

## Project Structure

.
├── cnn_cifar10_ablation.py
├── README.md

## Key Observations

* Learning rate strongly affects convergence speed and stability
* Increasing model depth improves feature learning but may increase overfitting
* Larger batch sizes improve training efficiency but may affect generalization
* Feature maps demonstrate hierarchical feature extraction in CNNs


