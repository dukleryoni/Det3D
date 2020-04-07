import os, sys
sys.path.insert(0, "/home/image/ohs/Det3D")
import torch

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import nuscenes
from nuscenes.nuscenes import NuScenes

from det3d.torchie import Config
from det3d.datasets import build_dataset


# file_path="/home/image/ohs/Det3D/examples/cbgs/configs/cbgs_car_only_try.py"
# cfg = Config.fromfile(file_path)
# nusc_val = build_dataset(cfg.data.val)
# nusc = NuScenes(version=nusc_val.version, dataroot=str(nusc_val._root_path), verbose=True)


def example_to_device(example, device, non_blocking=False) -> dict:
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "fsaf_targets"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in ["fsaf_mg_targets"]:
            example_torch[k] = [[res.to(device, non_blocking=non_blocking) for res in res_class] for res_class in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


