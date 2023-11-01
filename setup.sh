#!/bin/bash

# Install Git LFS
git lfs install

# Clone the repository
git clone https://huggingface.co/bert-base-uncased

# Install the requirements
pip install -r requirements.txt

# Run the loader.py to get data from the dataset
python loader.py
