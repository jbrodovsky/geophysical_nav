#!/bin/bash

# This script is used to build the project

# Note: Due to some sort of issue with environment variables, the script must be run with the source command:
# source builder.sh
# --- NOT ----
# ./builder.sh OR bash builder.sh

rm -r dist
rm -r sdist
rm -r build

python -m build
# find the .whl file under ./dist
WHL_FILE=$(find ./dist -name "*.whl")
# install the .whl file
python -m pip install $WHL_FILE --force-reinstall
