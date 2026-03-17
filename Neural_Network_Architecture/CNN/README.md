# CNN on CIFAR-10 (Training + Validation)

## Project Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch and includes both training and validation steps.

## Dataset
CIFAR-10 dataset (loaded using torchvision)

## Workflow

1. Data preprocessing using transforms
2. Training CNN model
3. Evaluating accuracy on test data
4. Adding validation loss tracking

## Architecture

Conv → ReLU → MaxPool  
Conv → ReLU → MaxPool  
Conv → ReLU → MaxPool  
Fully Connected Layer  
Output Layer (10 classes)

## Features Added (Day 5)

- Training loop
- Validation loop
- Loss tracking (Train vs Validation)

## Tools Used

- Python
- PyTorch
- Torchvision
- Jupyter Notebook