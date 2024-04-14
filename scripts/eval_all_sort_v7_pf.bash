#!/bin/bash

# Specify the folder path containing the AVI videos
folder="./results/yolov7/sort-pf/"

# Iterate over each AVI video in the folder
for file in "$folder"/*.txt; do
    python3 eval.py -r $file
done
