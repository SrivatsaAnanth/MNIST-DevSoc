# MNIST-DevSoc

This repository contains a Python implementation of a three-layer neural network for classifying handwritten digits from the MNIST dataset.

**Overview**
The model is implemented in prac4_final.py and uses a feedforward neural network with the following architecture:

Input Layer: 784 neurons (28x28 pixel images, flattened).

Hidden Layer 1: 128 neurons with leaky ReLU activation.

Hidden Layer 2: 64 neurons with leaky ReLU activation.

Output Layer: 10 neurons (one for each digit, 0–9) with softmax activation.

The model is trained using mini-batch gradient descent with a cross-entropy loss function. It processes the MNIST dataset (train.csv and test.csv) to train and generate predictions.

**Files**

prac4_final.py: Main script containing the neural network implementation, training, and testing logic.

train.csv: MNIST training dataset (60,000 images with labels).

test.csv: MNIST test dataset (10,000 images without labels).

**Requirements**

Python 3.x

Libraries: numpy, pandas, matplotlib

**Usage**

Ensure train.csv and test.csv are in the same directory as prac4_final.py.

Run the script: python prac4_final.py

The script will:

Train the model for 15 epochs with a batch size of 64 and learning rate of 0.1.
Display training accuracy and loss every 5 epochs.
Visualize a sample test image (index 83) and its predicted digit.

**Model Details**

Initialization: Weights are initialized using He initialization to optimize training with ReLU-like activations.

Training: Uses mini-batch gradient descent with shuffling to improve generalization.

Activations: Leaky ReLU (alpha=0.01) for hidden layers, softmax for output.
