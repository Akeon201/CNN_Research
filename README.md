# CNN Research

## Author
Kenyon LeBlanc

## Special Thanks
Dr. Girard

## Overview
This repository contains the code and resources used in the research on convolutional neural networks (CNNs). It includes the following:
- Python scripts for data analysis, model configuration, and testing.
- Requirements for setting up the Python environment.
- Processed and raw test results.
- Research paper and additional resources summarizing the findings.

## Requirements
- Python 3.11

### Installation
To set up the required environment, run:



# File Structure

- **code/**: Contains Python scripts used for analysis and testing.
  - **analysis.py**: Script for analyzing the results.
  - **config.py**: Configuration file for model parameters and settings.
  - **main.py**: Main script to run the CNN experiments.
  - **tests.py**: Unit tests for the code.
- **test_results_analyzed/**: Directory containing processed test results.
- **test_results_raw/**: Directory with raw test data.
- **results.csv**: Consolidated test results in CSV format.
- **requirements.txt**: Dependencies needed for the project.
- **LICENSE**: License details for the repository.
- **PowerPoint.pptx**: Presentation summarizing the research findings.
- **Research_Paper.pdf**: Detailed research paper.
- **README.md**: This README file.

## Configurations and Settings

The experiment settings are defined in the `config.py` file within the **code** folder. Below is an explanation of the key settings:

### General Settings
- **load_file**: Specifies whether to load existing results ('Y' or 'N').
- **path**: Path to the results folder (e.g., "tests/5_relu_1").

### Network Variables
- **num_conv_pool_layers**: Number of convolutional and pooling layers.
- **conv_activation**: Activation function in the convolution layer (e.g., 'leaky_relu').
- **num_neurons**: Number of neurons in the fully connected layer.
- **num_epochs**: Number of epochs for training.
- **weight_scale**: Scale of the initial weights.

### Learning Settings
- **test_num**: Test identifier; must be renamed for each unique test run.
- **learning_rates**: List of learning rates to experiment with.
- **learning_rates_str**: String representation of the learning rates.
- **lr_index**: Index to select a learning rate from `learning_rates`.
- **learning_rate**: Automatically set based on `lr_index`.
- **activ_str**: Activation function for the fully connected layer (e.g., 'relu', 'tanh', 'sigmoid', 'leaky_relu').
- **test_name**: Automatically generated test name based on the configuration.

### Classification Settings
- **fc_activation**: Activation function for fully connected layers.
- **classes_to_train**: List of class indices to include in training (e.g., [2, 3, 4, 5, 6, 7] for animals).

### Class Labels:
- 0: Airplane
- 1: Automobile
- 2: Bird
- 3: Cat
- 4: Deer
- 5: Dog
- 6: Frog
- 7: Horse
- 8: Ship
- 9: Truck

- **neither_label**: Whether to classify non-animal classes as "neither" ('Y' or 'N').

**Note: This cnn implementation uses the animal classes.**

### Data Limits
- **train_num**: Number of training samples (max: 30,000).
- **test_num**: Number of testing samples (max: 6,000).

## Research Paper
You can find the detailed research paper in the repository: **Research_Paper.pdf**.

## Results
The consolidated results of the CNN experiments can be found in **results.csv**.
