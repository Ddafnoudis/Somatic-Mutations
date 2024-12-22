#!/bin/bash

# Define the directories
DATA_DIR="/data"
OUTPUT_DIR="/test_dataset"

cd ../
# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

cd $DATA_DIR

for tarfile in *.tar.gz; do
  # Extract the data_mutations.txt file from the tar file
  tar -xf "${tarfile}"
  # Define the folder's name
  folder_name="$(basename "${tarfile%.tar.gz}")"
  # Get into the tar folder of the tar file
  cd $folder_name
  # Check if the data_mutations.txt file exists
  # if [ -f data_mutations.txt ]; then
  # Rename the data_mutations.txt file into the tar file basename 
  echo "Renaming data_mutations.txt to ${tarfile%.tar.gz}_data_mutations.txt"
  mv "data_mutations.txt" "${tarfile%.tar.gz}_data_mutations.txt"
  # Move the data_mutations.txt file to the output directory
  mv "${tarfile%.tar.gz}_data_mutations.txt" "$OUTPUT_DIR"
  echo "Moved ${tarfile%.tar.gz}_data_mutations.txt to $OUTPUT_DIR"
  rm -r $folder_name
  cd ../

done
