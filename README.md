# SORT + PF (Particle Filter)

This code is a Python implementation of the SORT + PF (Particle Filter) algorithm for object tracking in challenging environments.


## Installation

To install the required packages, run the following command:

> Note: Python 3.7 is required to run this code due to libraries conflicts.

```bash
pip install -r requirements.txt
```

## Dataset

The code was tested on the [UAVPorpoise dataset](https://putvision.github.io/UAVPorpoises/).

## Usage

To run the code, use the following command:

```bash
python3 main.py --task viz --tracker sort-pf --source ./dataset/test_videos/000.avi --engine cpu --cache-yolo
```

## Evaluation

To evaluate the tracker, use the following command:

```bash
bash ./scripts/track_all.bash <tracker> <motion_model>
```

- available trackers: `base` (SORT), `pf` (SORT + PF)
- available motion models: `none`, `flow` (Sparse Optical Flow, aka SOF)

Then run the following command to evaluate the tracker:

```bash
python3 eval_track.py
```

## Results

| Method         | HOTA :arrow_up: | ID-SW :arrow_down: | ID-F1 :arrow_up: |
|----------------|:----------------:|:-------------------:|:-----------------:|
| Strong-SORT    |       0.514      |        10.600       |       0.535       |
| OC-SORT        |       0.526      |        3.400   |       0.597       |
| BoT-SORT       |       0.580      |        8.633        |       0.628       |
| BoT-SORT (ECC) |       0.451      |        32.833       |       0.466       |
| BoT-SORT (ORB) |       0.454      |        4.533        |       0.495       |
| BoT-SORT (SOF) |       0.604      |        4.400        |       0.650       |
| SORT           |       0.568      |        4.533        |       0.627       |
| SORT (SOF)     |       0.568      |        4.567        |       0.628       |
| SORT-PF        |       0.657                  |        3.567        |      :fire: 0.699 :fire:      |
| SORT-PF (SOF)  |      :fire: 0.660 :fire:     |       :fire: 3.300 :fire:       |  0.687  |
