#!/bin/bash
# Generates all hierarchies
set -e

# Ensure we are in the project root (simple check)
if [ ! -f "app.py" ]; then
    echo "Please run this script from the project root directory."
    exit 1
fi

mkdir -p output

bash scripts/generate_imagenet.sh
bash scripts/generate_coco.sh
bash scripts/generate_openimages.sh

echo "All hierarchies generated in output/"
