#!/bin/bash

# Get the input file location for raw data
echo "This script will parse the NOAA dataset and prepare it for use with the NAV package. It parses raw NOAA mgd77 data in two steps. First it reads in and cleans up the dataset into a continuous data stream where each point has a bathymetric depth, magnetic anomally, and gravity anomaly measured. Then it parses the continuous data stream into segments of continuous data. The script will ask for the location of the raw data file, the output file location for the processed data, and output location for the parsed segments, and the parameters for parsing the data."
echo "Enter the location of the raw data file:"
read raw_data_file
echo "Enter the output file location for the processed data:"
read processed_data_file
echo "Enter the output file location for the parsed segments:"
read parsed_segments_file
echo "Now parsing the data. Enter the maximum time between two points to mark a break in continuous data (time in minutes):"
read max_time
echo "Enter the maximum time between two points in a continuous segment to validate the segment (time in minutes):"
read max_delta_t
echo "Enter the minimum duration of a continuous segment to validate the segment (time in minutes):"
read min_duration
# Inital clean up of the data
python process_dataset.py --mode mgd77 --location $raw_data_file --output $processed_data_file
# Run the parsing
python process_dataset.py --mode parser --location $processed_data_file --output $parsed_segments_file --max_time $max_time --max_delta_t $max_delta_t --min_duration $min_duration