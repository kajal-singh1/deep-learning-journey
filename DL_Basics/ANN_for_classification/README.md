# Date Fruit Classification using Neural Network

## Project Overview
This project implements an Artificial Neural Network (ANN) using PyTorch to classify different types of date fruits based on their physical characteristics.

## Dataset
Dataset: Date Fruit Dataset

Features include measurements such as:
- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Eccentricity
- Convex Area
- Extent
- Solidity
- Roundness
- Compactness
- Shape Factor

Target:
- Class (Type of Date Fruit)

## Project Workflow

1. Data Loading using Pandas
2. Feature and target separation
3. Label Encoding for class labels
4. Train-Test Split
5. Feature Scaling using StandardScaler
6. Neural Network training using PyTorch
7. Model evaluation using accuracy

## Neural Network Architecture

Input Layer: Number of dataset features  
Hidden Layer 1: 64 neurons + ReLU  
Hidden Layer 2: 64 neurons + ReLU  
Output Layer: 7 neurons (date fruit classes)

Loss Function:
CrossEntropyLoss

Optimizer:
Adam

Epochs:
100

## Tools Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Output

The model predicts the class of date fruits and evaluates performance using classification accuracy.