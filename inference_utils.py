import os, sys
sys.path.insert(0, "/home/image/ohs/Det3D")
import torch
import operator
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

from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)

#
# general_to_detection = {
#     "human.pedestrian.adult": "pedestrian",
#     "human.pedestrian.child": "pedestrian",
#     "human.pedestrian.wheelchair": "ignore",
#     "human.pedestrian.stroller": "ignore",
#     "human.pedestrian.personal_mobility": "ignore",
#     "human.pedestrian.police_officer": "pedestrian",
#     "human.pedestrian.construction_worker": "pedestrian",
#     "animal": "ignore",
#     "vehicle.car": "car",
#     "vehicle.motorcycle": "motorcycle",
#     "vehicle.bicycle": "bicycle",
#     "vehicle.bus.bendy": "bus",
#     "vehicle.bus.rigid": "bus",
#     "vehicle.truck": "truck",
#     "vehicle.construction": "construction_vehicle",
#     "vehicle.emergency.ambulance": "ignore",
#     "vehicle.emergency.police": "ignore",
#     "vehicle.trailer": "trailer",
#     "movable_object.barrier": "barrier",
#     "movable_object.trafficcone": "traffic_cone",
#     "movable_object.pushable_pullable": "ignore",
#     "movable_object.debris": "ignore",
#     "static_object.bicycle_rack": "ignore",
# }
#
# for n in cfg.class_names:
#     if n in inference_utils.general_to_detection:
#         mapped_class_names.append(inference_utils.general_to_detection[n])
#     else:
#         mapped_class_names.append(n)


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



def get_nusc_style(predictions, nusc, output_dir=None, testset=False):
    mapped_class_names = ['car',
                          'truck',
                          'construction_vehicle',
                          'bus',
                          'trailer',
                          'barrier',
                          'motorcycle',
                          'bicycle',
                          'pedestrian',
                          'traffic_cone']

    nusc_annos = {
      "results": {},
      "meta": None,}

    for prediction in predictions:
        annos = []
        boxes = _second_det_to_nusc_box(prediction)
        boxes = _lidar_nusc_box_to_global(nusc, boxes, prediction["metadata"]["token"])

        for i, box in enumerate(boxes):
            name = mapped_class_names[box.label]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in [
                    "car",
                    "construction_vehicle",
                    "bus",
                    "truck",
                    "trailer",
                ]:
                    attr = "vehicle.moving"
                elif name in ["bicycle", "motorcycle"]:
                    attr = "cycle.with_rider"
                else:
                    attr = None
            else:
                if name in ["pedestrian"]:
                    attr = "pedestrian.standing"
                elif name in ["bus"]:
                    attr = "vehicle.stopped"
                else:
                    attr = None

            nusc_anno = {
                "sample_token": prediction["metadata"]["token"],
                "translation": box.center.tolist(),
                "size": box.wlh.tolist(),
                "rotation": box.orientation.elements.tolist(),
                "velocity": box.velocity[:2].tolist(),
                "detection_name": name,
                "detection_score": box.score,
                "attribute_name": attr
                if attr is not None
                else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                    0
                ],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({prediction["metadata"]["token"]: annos})

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    return nusc_annos