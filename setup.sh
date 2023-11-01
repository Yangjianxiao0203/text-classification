#!/bin/bash

#conda
# conda create --name nlp python=3.10
conda activate nlp

# Install the requirements
pip install -r requirements.txt

# Install Git LFS
# git lfs install

# Clone the repository
# git clone https://huggingface.co/bert-base-uncased

# Run the loader.py to get data from the dataset
python loader.py
