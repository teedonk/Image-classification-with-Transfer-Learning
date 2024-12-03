# Image-classification-with-Transfer-Learning
# Cat and Dog Classification

This repository contains a project for classifying images of cats and dogs using transfer learning. The solution is implemented in Python and leverages TensorFlow, Keras, and Google Colab for seamless training and evaluation.

# Overview

This project applies transfer learning techniques to classify images as either cats or dogs. It demonstrates the use of pre-trained models to enhance accuracy and reduce training time, while utilizing Google Colab for computational efficiency.

# Features

Transfer learning with pre-trained models (InceptionResNetV2).

Data organization for training, validation, and testing.

Performance visualization with loss and accuracy metrics.

Easy-to-run script using Google Colab.

# Requirements

Python 3.7+

TensorFlow 2.0+

Matplotlib

NumPy

Google Colab (for cloud execution)

pydrive (for dataset access)

Install dependencies with:
##
    pip install tensorflow matplotlib numpy pydrive

Getting Started

Clone the repository:

##
    git clone [https://github.com/teedonk/Image-classification-with-Transfer-Learning]


Open the notebook in Google Colab:
Cat and Dog Classification Notebook

Run the cells sequentially to download the dataset, train the model, and evaluate performance.

Dataset Preparation

The dataset is downloaded and organized automatically within the notebook. It consists of:

Training set: Images of cats and dogs.

Validation set: Used to tune the model.

Test set: Used to evaluate the final model.


# Model Architecture

The model utilizes a pre-trained convolutional neural network for feature extraction. A fully connected layer is added on top to perform binary classification. The script supports:

Fine-tuning the pre-trained model.

Batch normalization and dropout for regularization.

# Results

The model achieves high accuracy on the validation and test datasets, demonstrating the effectiveness of transfer learning for image classification tasks. Training and validation accuracy/loss plots are generated during training.

