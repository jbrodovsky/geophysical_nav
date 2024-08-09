#!/bin/bash

# Note: Due to some sort of issue with environment variables, the script must be run with the source command:
# source setup_env.sh NOT ./setup_env.sh OR bash setup_env.sh

# Set up the environment for the project by checking to see if miniconda is installed and if so activate the
# base environment and create the environment for the project. If miniconda is not installed, install it and
# create the environment for the project

# Check if miniconda is installed
if ! command -v conda &> /dev/null
then
    # Install miniconda
    echo -e "\e[31mMiniconda is not installed. Installing Miniconda...\e[0m"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
    conda init --all
    source ~/.bashrc
else
    echo -e "\e[32mMiniconda is already installed.\e[0m"
fi
# Create the environment for the project
echo "Creating the environment for the project..."
# if the environment already exists, remove it
if conda env list | grep -q geonav; then
    echo -e "\e[31mEnvironment already exists\e[0m"
    # ask if the user would like to rebuild the environment
    read -p "Would you like to rebuild the environment? (y/n): " rebuild
    if [ "$rebuild" != "y" ]; then
        echo "Exiting..."
        exit 0
    fi
    conda activate base
    conda env remove --name geonav -y
    echo "Existing environment removed. Rebuilding environment..."
fi
conda env create --file environment.yml --name geonav
conda activate geonav
echo "Installing addtional PIP-only packages..."
python -m pip install -r requirements.txt
# conda update --all -y
echo -e "\e[32mEnvironment setup complete.\e[0m"
