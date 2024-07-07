#!/bin/bash

# Set up the environment for the project by checking to see if miniconda is installed and if so activate the base environment and create the environment for the project
# If miniconda is not installed, install it and create the environment for the project

# Check if miniconda is installed
if ! command -v conda &> /dev/null
then
    # Install miniconda
    echo "Miniconda is not installed. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc
else
    echo "Miniconda is already installed."
fi

# Create the environment for the project
echo "Creating the environment for the project..."
conda env create -f environment.yml
conda init bash
conda activate nav
echo "Installing addtional PIP-only packages..."
python -m pip install requirements.txt
