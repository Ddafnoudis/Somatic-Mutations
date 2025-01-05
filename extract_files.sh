#!/bin/bash

# Define the directories
read -p "Enter the data directory: " DATA_DIR
read -p "Enter the output directory: " OUTPUT_DIR

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

cd $DATA_DIR || { echo "Failed to change directory to $DATA_DIR"; exit 1; }
set -x
for tarfile in *.tar.gz; do
  echo $tarfile
  # Extract the data_mutations.txt file from the tar file
  tar -xf "${tarfile}"
  
  # Define the folder's name
  folder_name="$(basename "${tarfile%.tar.gz}")"

  # Name of the folder
  echo "This is the folder name" $folder_name

  # Get into the tar folder of the tar file
  cd $folder_name

  # Check if the data_mutations.txt file exists
  if [ -f "data_mutations.txt" ]; then
    # Rename the data_mutations.txt file into the tar file basename 
    echo "Renaming data_mutations.txt to ${tarfile%.tar.gz}_data_mutations.txt"
    mv "data_mutations.txt" "${tarfile%.tar.gz}_data_mutations.txt"
    # Move the data_mutations.txt file to the output directory
    echo "Moving ${tarfile%.tar.gz}_data_mutations.txt to $OUTPUT_DIR"
    mv "${tarfile%.tar.gz}_data_mutations.txt" "$OUTPUT_DIR"
    echo -e "${tarfile}_data_mutations.txt finished!\n\n"
  # # Move back
  # cd ../
  # Finished loop
  fi
set -x
  if [ -f "data_clinical_sample.txt" ]; then
    # Rename data_clinical_sample into the tar file basename
    echo "Renaming data_clinical_sample to ${tarfile%.tar.gz}_data_clinical_sample.txt"
    mv "data_clinical_sample.txt" "${tarfile%.tar.gz}_data_clinical_sample.txt"
    echo "Moving ${tarfile%.tar.gz}_data_clinical_sample.txt to $OUTPUT_DIR"
    mv "${tarfile%.tar.gz}_data_clinical_sample.txt" "$OUTPUT_DIR"
    echo -e "${tarfile}_data_clinical_sample.txt finished!\n\n"
  cd ../
  fi

  # Navigate to the data directory
cd "$DATA_DIR" || { echo "Failed to move to $DATA_DIR"; exit 1; }

# Print the current working directory after changing to data directory
echo "Current working directory after cd to DATA_DIR: $(pwd)"

# Remove the folder
if [ -d "$folder_name" ]; then
    echo "Removing folder: $folder_name"
    rm -r "$folder_name" || { echo "Failed to remove directory $folder_name"; exit 1; }
else
    echo "Directory $folder_name does not exist"
fi

done

