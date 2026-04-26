# Handwritten-Digit-Recognizer-CNN-
This project builds a Convolutional Neural Network (CNN) to recognize handwritten digits from the famous MNIST dataset.
# Handwritten Digit Recognizer ✍️

A Deep Learning project that uses a Convolutional Neural Network (CNN) to classify images of handwritten digits (0-9). 

## Overview
This project was built using **TensorFlow and Keras**. It trains a multi-layer CNN on the classic **MNIST dataset**, which contains 60,000 training images and 10,000 testing images of handwritten numbers. The model achieves over 98% accuracy on unseen test data.

## Tech Stack
* Python 3.x
* TensorFlow / Keras (Deep Learning Framework)
* NumPy (Matrix operations)
* Matplotlib (Data visualization)

## Model Architecture
* **Conv2D Layer** (32 filters) + ReLU activation
* **MaxPooling2D Layer**
* **Conv2D Layer** (64 filters) + ReLU activation
* **MaxPooling2D Layer**
* **Conv2D Layer** (64 filters) + ReLU activation
* **Flatten Layer**
* **Dense Layer** (64 neurons)
* **Dense Output Layer** (10 neurons, Softmax activation)

## How to Run Locally
1. Clone this repository.
2. Install the necessary Python packages:
   ```bash
   pip install tensorflow matplotlib numpy
