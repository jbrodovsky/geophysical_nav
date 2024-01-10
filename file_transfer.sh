#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source_path> <destination_user@host> <destination_path>"
    exit 1
fi

SOURCE_PATH=$1
DESTINATION=$2
DESTINATION_PATH=$3

# Check if the source path is a directory
if [ -d "${SOURCE_PATH}" ]; then
    SOURCE_PATH="${SOURCE_PATH}/"
fi

# Use scp to transfer the file or directory
scp -r "${SOURCE_PATH}" "${DESTINATION}:${DESTINATION_PATH}"