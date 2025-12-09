#!/bin/bash
# Generates ImageNet hierarchy (filtered to 1k classes, deep recursion)
echo "Generating ImageNet Hierarchy..."
python app.py imagenet-tree --root entity.n.01 --depth 25 --filter --output output/imagenet.yaml
