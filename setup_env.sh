#!/bin/bash

# Note: Due to some sort of issue with environment variables, the script must be run with the source command:
# source setup_env.sh NOT ./setup_env.sh OR bash setup_env.sh

# Set up the environment for the project by checking to see if pixi is installed and installing it if it is not

# Check if pixi is installed
if ! command -v pixi &> /dev/null
then
    echo "Pixi is not installed. Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
else
    echo "Pixi is already installed. Sourcing environment..."
    pixi shell
fi