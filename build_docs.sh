
#!/bin/bash

# Run this file by using the command `bash -i build_docs.sh`

source ~/.bashrc

# Activate the conda environment 'nav'
conda activate nav

# Navigate to the /doc/ folder
cd ./doc/

# Run the command to build the html documentation using Sphinx
make html
