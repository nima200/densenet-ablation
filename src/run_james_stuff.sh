#!/usr/bin/env bash

# Activate Anaconda VirtualEnv
source activate base

# Run Composite Function Permutations
python3 cifar10/CompositeFunctions/cifar10.py

# Run Regularization tests
python3 cifar10/FashionMNIST/fashion_mnist.py

#Run Fashion MNIST
python3 FashionMNIST/fashion_mnist.py



