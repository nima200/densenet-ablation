#!/usr/bin/env bash

# Activate Anaconda VirtualEnv
source activate base

# Install dependencies from requirements
pip3 install -r requirements.txt
conda install -r requirements.txt
