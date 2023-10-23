---

# Handwritten Digits Classification

This project is a Handwritten Digits Classification system implemented using TensorFlow. It includes data loading, pre-processing, model training, testing, and saving, as well as making predictions using the trained model.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Testing and Evaluation](#testing-and-evaluation)
9. [Sample Prediction](#sample-prediction)
10. [Results](#results)
11. [Saved Model](#saved-model)
12. [Issues](#issues)
13. [References](#references)

## Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- Pillow (PIL)

## Project Structure

The project's structure includes the following files:
- `train.py`: The main script for training the handwritten digits classification model.
- `test.py`: A script for testing the trained model and making predictions.
- `MNIST/`: A directory containing the MNIST dataset and saved models.
- `SavedModel-512-64 LR0.001 E18/`: A directory containing a saved model.

## Installation

To set up your environment, you can install the required libraries using pip:

```bash
pip install tensorflow numpy pillow
```

## Usage

1. Training:
   - Run the `Neural Networks - TF handwriting Recognition.ipynb` script to train the model. Modify hyperparameters and settings as needed.
   - The model will be saved in the specified directory.

2. Testing and Prediction:
   - Use the `SavedModel Load.ipynb` script to test the model on new data and make predictions.
   - Load the trained model using the provided path.

## Data

The dataset used in this project is the MNIST dataset, which contains hand-drawn digits.

## Model Architecture

The neural network architecture consists of two hidden layers:
- Hidden Layer 1: 512 units with ReLU activation
- Hidden Layer 2: 64 units with ReLU activation
- Output Layer: Softmax activation with 10 classes

## Training

- The training process includes 18 epochs.
- Adam optimizer with a learning rate of 0.001 is used.
- Training data is split into training and validation sets.

## Testing and Evaluation

- After training, the model's accuracy is tested on the provided test dataset.
- The test dataset prediction accuracy is reported.

## Sample Prediction

A sample image (`test_img.png`) is provided to make predictions using the trained model.

## Results

The model achieved an accuracy of 97.64% on the test dataset.

## Saved Model

The trained model is saved in the `SavedModel-512-64 LR0.001 E18/` directory.

## Issues

If you encounter any issues or have questions, please create an issue on the GitHub repository.

## References

- [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database)
- [TensorFlow Documentation](https://www.tensorflow.org/)


---
