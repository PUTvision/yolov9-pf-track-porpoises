#!/bin/bash

# Specify the folder path containing the AVI videos
folder="./dataset/test_videos/"
tracker_name="sort"
model_path="./data/best-yolov9.onnx"

# $1 have to be base or pf

if [ $1 == "base" ]; then
    tracker_name+=""
elif [ $1 == "pf" ]; then
    tracker_name+="-pf"
else
    echo "Invalid parameter: $1 (base or pf)"
    exit 1
fi

# $2 have to be warp, sensor or none

if [ $2 == "flow" ]; then
    tracker_name+="-flow"
elif [ $2 == "none" ]; then
    tracker_name+=""
else
    echo "Invalid parameter: $2 (flow, sensor or none)"
    exit 1
fi

echo $tracker_name


# # Remove the previous results if exist
rm -rf "./track_data/trackers/MOT17-test/"$tracker_name

# Iterate over each AVI video in the folder
for file in "$folder"/*.avi; do
    python3 main.py --task pred --tracker $tracker_name --source $file --weights $model_path --engine cpu --cache-yolo
done
