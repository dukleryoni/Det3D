# Det3D

A general 3D Object Detection codebase in PyTorch

## Introduction

Det3D is the first 3D Object Detection toolbox which provides off the box implementations of many 3D object detection algorithms such as PointPillars, SECOND, PIXOR, etc, as well as state-of-the-art methods on major benchmarks like KITTI(ViP) and nuScenes(CBGS). Key features of Det3D include the following aspects:

* Multi Datasets Support: KITTI, nuScenes, Lyft
* Point-based and Voxel-based model zoo
* State-of-the-art performance
* DDP & SyncBN

## Installation

Please refer to [INSTALL.md](INSTALL.md).

## Quick Start

Please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## Model Zoo and Baselines

### [Second](examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py) on KITTI(val) Dataset

```
car  AP @0.70, 0.70,  0.70:
bbox AP:90.66, 89.30, 88.38
bev  AP:89.90, 87.69, 86.67
3d   AP:88.33, 78.03, 76.81
aos  AP:90.45, 88.80, 87.72
```

### [PointPillars](examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py) on KITTI(val) Dataset

```	
car  AP@0.70,  0.70,  0.70:
bbox AP:90.69, 88.76, 87.34
bev  AP:89.64, 86.22, 83.14
3d   AP:86.65, 76.15, 69.34
aos  AP:90.61, 88.34, 86.65
```
### [PointPillars](examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn.py) on NuScenes(val) Dataset

```
mAP: 0.3031
mATE: 0.4472
mASE: 0.2755
mAOE: 0.9103
mAVE: 0.7139
mAAE: 0.2472
NDS: 0.3922

car Nusc dist AP@0.5, 1.0, 2.0, 4.0
58.75 73024 78.36 81.46 mean AP: 0.7295
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
14.67 31.79 41.26 45.67 mean AP: 0.3336
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00 0.51 2.87 5.39 mean AP: 0.0219
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
11.17 32.66 51.35 60.46 mean AP: 0.3891
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
1.13 12.69 29.28 39.60 mean AP: 0.2068
barrier Nusc dist AP@0.5, 1.0, 2.0, 4.0
14.12 37.58 47.66 52.82 mean AP: 0.3804
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
8.32 11.40 13.03 13.75 mean AP: 0.1162
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.03 0.08 0.11 0.20 mean AP: 0.0010
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
44.40 55.88 58.51 61.47 mean AP: 0.5506
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
24.37 27.11 31.16 38.27 mean AP: 0.3023
```

### [CGBS](examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on NuScenes(val) Dataset
```
mAP: 0.4669
mATE: 0.3391
mASE: 0.2574
mAOE: 0.7657
mAVE: 0.3162
mAAE: 0.2012
NDS: 0.5455

car Nusc dist AP@0.5, 1.0, 2.0, 4.0
68.54, 80.63, 84.63, 86.58 mean AP: 0.8009424366048317
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
24.87, 44.61, 55.15, 58.78 mean AP: 0.45850578704319195
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.17, 6.90, 16.21, 24.29 mean AP: 0.11890247733120854
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
32.94, 56.79, 73.28, 76.13 mean AP: 0.5978476967252525
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
4.21, 20.78, 37.94, 53.70 mean AP: 0.29155996509713616
barrier Nusc dist AP@0.5, 1.0, 2.0, 4.0
39.77, 55.13, 59.95, 62.47 mean AP: 0.5433129255795658
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
31.97, 39.27, 40.01, 40.37 mean AP: 0.3790460790422441
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
13.72, 14.36, 14.47, 14.65 mean AP: 0.1430066611617972
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
72.81, 75.17, 76.99, 78.74 mean AP: 0.7592805073941438
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
53.54, 55.51, 58.53, 63.15 mean AP: 0.5768369346063308
```
### To Be Released

3. [CGBS](examples/cbgs/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on Lyft(val) Dataset (too many bugs. got 0 accuracy)

## Currently Support

* Models
  - [x] VoxelNet
  - [x] SECOND
  - [x] PointPillars
* Features
    - [x] Multi task learning & Multi-task Learning
    - [x] Distributed Training and Validation
    - [x] SyncBN
    - [x] Flexible anchor dimensions
    - [x] TensorboardX
    - [x] Checkpointer & Breakpoint continue
    - [x] Self-contained visualization
    - [x] Finetune
    - [x] Multiscale Training & Validation
    - [x] Rotated RoI Align


## TODO List
* Models
  - [ ] PointRCNN
  - [ ] PIXOR

## Developers

[Benjin Zhu](https://github.com/poodarchu/) , [Bingqi Ma](https://github.com/a157801)

## License

Det3D is released under the [MIT licenes](LICENES).

## Acknowlegement

* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
