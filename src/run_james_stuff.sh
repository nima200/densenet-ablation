#!/usr/bin/env bash

# Activate Anaconda VirtualEnv
source activate base

# Run Composite Function Permutations
python3 cifar10/CompositeFunctions/cifar10.py true 2>&1 | tee -a cifar10/CompositeFunctions/log

# Run Regularization tests
python3 cifar10/Regularization/cifar10.py true 2>&1 | tee -a cifar10/Regularization/log

#Run Fashion MNIST
python3 FashionMNIST/fashion_mnist.py true 2>&1 | tee -a FashionMNIST/log



