#!/bin/bash

# Specify the folder path containing the AVI videos
folder="./dataset/test_videos/"
tracker_name="botsort"
model_path="./data/best-yolov9.onnx"


for cmc_method in "sparseOptFlow"; do
    # Remove the previous results if exist
    rm -rf "./track_data/trackers/MOT17-test/"$tracker_name"-"$cmc_method

    # Iterate over each AVI video in the folder
    for file in "$folder"/*.avi; do
        python3 main.py --task pred --tracker $tracker_name"-"$cmc_method --source $file --weights $model_path --engine cpu --cache-yolo --disable-keypoints
    done
done
