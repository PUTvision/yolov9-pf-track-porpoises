#!/bin/bash

# Specify the folder path containing the AVI videos
folder="../dataset/test_videos/"

# Iterate over each AVI video in the folder
for file in "$folder"/*.avi; do
    python3 main.py --task viz --tracker sort --source $file --model-path ./data/best.onnx --engine cuda
done
