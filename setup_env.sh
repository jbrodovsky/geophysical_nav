#!/bin/bash

# Note: Due to some sort of issue with environment variables, the script must be run with the source command:
# source setup_env.sh
# --- NOT ----
# ./setup_env.sh OR bash setup_env.sh

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
    conda init --all
    source ~/.bashrc
else
    echo "Miniconda is already installed."
fi

# Create the environment for the project
echo "Creating the environment for the project..."
conda env create --file environment.yml --name geonav
conda activate geonav
echo "Installing addtional PIP-only packages..."
python -m pip install -r requirements.txt
conda update --all -y
