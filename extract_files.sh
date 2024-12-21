#!/bin/bash

# Set variables for directories
DATA_DIR="data"
OUTPUT_DIR="datasets"

# Create the datasets directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Navigate to the data directory
cd "$DATA_DIR"

# Extract the tar.gz file
tar -xzvf *.tar.gz*
# Find the nested mutation file and move it to the datasets directory with the renamed file
mutation_file=$(find . -name "data_mutations.txt")
if [ -n "$mutation_file" ]; then
    mv "$mutation_file" "../../$OUTPUT_DIR/${base_name}_mutations.txt"
else
    echo "No mutation file found in $base_name."
fi

