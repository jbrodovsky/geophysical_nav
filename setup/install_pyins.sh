#!/bin/bash

# check to see if a virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "A virtual environment is not currently active"
    # Prompt the user to create a v
    read -p "Would you like to create a local virtual environment? (y/n) " response
    if [ "$response" = "y" ]; then
        # Create a virtual environment
        python3 -m venv venv
        # Activate the virtual environment
        source venv/bin/activate
        pip install --upgrade pip
    else
        echo "Please activate a virtual environment before running this script"
        exit 1
    fi
fi

# Clone the pyins repository
echo "Cloning the pyins repository"
git clone https://github.com/nmayorov/pyins.git

# Navigate into the pyins directory
cd pyins

# Install pyins with pip
echo "Installing pyins"
pip install .

# Run the tests to ensure everything is working correctly
echo "Running tests"
pytest

# Navigate back to the original directory
echo "Removing the pyins directory"
cd ..
rm -rf pyins

echo "pyins has been successfully installed"
