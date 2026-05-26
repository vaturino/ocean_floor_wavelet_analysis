#!/bin/bash

# Activate the environment
source /Users/valeturino/postdoc_local/wavelet_analysis/bin/activate

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
TARGET_DIR="/Users/valeturino/postdoc_local/wavelet_analysis"
JSON_DIR="/Users/valeturino/postdoc_local/seafloor_json_mapping_test"
PYTHON_SCRIPT="automatic_global_data_gaussian_mixture.py"

# Validate that the JSON directory exists
if [ ! -d "$JSON_DIR" ]; then
    echo "Error: Directory $JSON_DIR does not exist." >&2
    exit 1
fi

# Move into the JSON directory so we are working with relative/local paths
cd "$JSON_DIR"

# Loop through all .json files in the current directory
for file in *.json; do
    # Safeguard: Ensure files actually exist matching the glob
    [ -e "$file" ] || continue
    
    # Strip the .json extension
    clean_name="${file%.json}"    
    echo "Processing file: $clean_name (from $file)"
    
    # Run the python script using its absolute path, passing the name without suffix
    python3 "$TARGET_DIR/$PYTHON_SCRIPT" "$clean_name"
done

echo "All files processed successfully."