#!/usr/bin/env bash

# Activate Anaconda VirtualEnv
source activate base

# Install dependencies from requirements
pip install -r requirements.txt
conda install --yes --file requirements.txt
