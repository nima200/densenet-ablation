#!/usr/bin/env bash

# Activate Anaconda VirtualEnv
source activate base

# Run Composite Function Permutations
#python3 cifar10/CompositeFunctions/cifar10.py 2>&1 | tee -a cifar10/CompositeFunctions/log

# Run Dropout tests
python3 cifar10/Dropout/cifar10.py 2>&1 | tee -a cifar10/Dropout/log

# Run Regularization tests
python3 cifar10/Regularization/cifar10.py 2>&1 | tee -a cifar10/Regularization/log

#Run Fashion MNIST
python3 FashionMNIST/fashion_mnist.py 2>&1 | tee -a FashionMNIST/log



