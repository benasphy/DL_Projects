# CNN Model Performance Analysis

This project provides a comprehensive analysis of Convolutional Neural Network (CNN) performance on the CIFAR-10 image classification dataset.

## Features

- **Multiple CNN Architectures**: Train and compare standard, deep, and lightweight CNN models
- **Interactive Visualizations**: Analyze model performance with interactive charts and visualizations
- **Data Augmentation**: Apply various data augmentation techniques to improve model performance
- **Feature Map Visualization**: Explore how CNNs "see" images through feature map visualization
- **Detailed Performance Metrics**: Evaluate models with accuracy, confusion matrices, and per-class analysis

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Streamlit

### Installation

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```

### Running the Application

```
streamlit run app.py
```

## Usage

1. Configure your CNN model in the sidebar:
   - Select model architecture (standard, deep, or lightweight)
   - Enable/disable data augmentation
   - Set training epochs and batch size

2. Click "Train CNN Model" to train the model

3. Explore the different tabs to analyze model performance:
   - Training History: View accuracy and loss curves
   - Accuracy Analysis: Examine per-class accuracy metrics
   - Confusion Matrix: Identify common misclassifications
   - Feature Maps: Visualize what the CNN "sees" in different layers
   - Common Mistakes: Analyze misclassified examples

## Model Architectures

### Standard CNN
A balanced architecture with 3 convolutional blocks, batch normalization, and dropout.

### Deep CNN
A deeper architecture with more layers and filters, suitable for more complex tasks.

### Lightweight CNN
A simpler architecture with fewer parameters, ideal for faster training and deployment.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- 50,000 training images
- 10,000 testing images

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
