#!/bin/bash

# Specify the folder path containing the AVI videos
folder="./dataset/test_videos/"
tracker_name="sort-pf-flow"
model="yolov9"
model_path="./data/best-yolov9.onnx"

# Remove the previous results if exist
rm -rf "./track_data/trackers/MOT17-test/"$model"_"$tracker_name


# Iterate over each AVI video in the folder
for file in "$folder"/*.avi; do
    python3 main.py --task pred --tracker $tracker_name --source $file --model $model --model-path $model_path --engine cpu --cache-yolo --disable-keypoints
done
