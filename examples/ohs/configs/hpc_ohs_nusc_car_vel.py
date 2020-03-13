import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["car"],),
  #  dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.6, 3.9, 1.56],
            anchor_ranges=[-50.4, -50.4, -0.95, 50.4, 50.4, -0.95], # Different than Claire's KITTI [0, -32.0, -1.00, 52.8, 32.0, -1.00]
            rotations=[0, 1.57],
            velocities=[0, 0],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="car",
        ),

        # dict(
        #     type="anchor_generator_range",
        #     sizes=[0.67, 0.73, 1.77],
        #     anchor_ranges=[-50.4, -50.4, -0.935, 50.4, 50.4, -0.935],
        #     rotations=[0, 1.57],
        #     velocities=[0, 0],
        #     matched_threshold=0.6,
        #     unmatched_threshold=0.4,
        #     class_name="pedestrian",
        # ),
        # dict(
        #     type="anchor_generator_range",
        #     sizes=[0.41, 0.41, 1.07],
        #     anchor_ranges=[-50.4, -50.4, -1.285, 50.4, 50.4, -1.285],
        #     rotations=[0, 1.57],
        #     velocities=[0, 0],
        #     matched_threshold=0.6,
        #     unmatched_threshold=0.4,
        #     class_name="traffic_cone",
        # ),

    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=9, linear_dim=False, encode_angle_vector=False,
)

# model settings
model = dict(
    type="VoxelNet_OHS",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=5,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(
        type="SpMiddleFHD", num_input_features=5, ds_factor=8, norm_cfg=norm_cfg,
    ),
    # neck=dict(
    #     type="RPN",
    #     layer_nums=[5,],
    #     ds_layer_strides=[1,],
    #     ds_num_filters=[128,],
    #     us_layer_strides=[1,],
    #     us_num_filters=[128,],
    #     num_input_features=128,
    #     norm_cfg=norm_cfg,
    #     logger=logging.getLogger("RPN"),
    # ),

    neck=dict(
        type="RPNV2",
        layer_nums=[5, ],
        layer_strides=[1, ],
        num_filters=[128, ],
        upsample_strides=[1, ],
        num_upsample_filters=[128, ],
        num_input_features=128,
        num_groups=32,
        use_norm=True,
        use_groupnorm=False,
        # norm_cfg=norm_cfg,
        logger=logging.getLogger("RPNV2"),

        ohs=dict(
            tasks=tasks,
            fsaf=1,
            encode_background_as_zeros=True,
            use_iou_branch=False,
            iou_loss_weight=1,
            eval_fsaf=1,
            fsaf_module=dict(
                use_border=False,
                norm_factor=8.0,
                limit_points=False,
                num_points=128,
                feat_strides=[8],
                loc_type="center_softbin",
                center_range=3,
                center_bin_num=12,
                rot_type="cos_sin",
                h_loc_type="softbin",
                dim_type="log",
                h_bin_num=16,
                rot_bin_num=36,
                s1=0.7,
                s2=1.5, # changed due to sparsity from (s2,s3) = 0.9, 1.0 in KITTI
                s3=2.0,
                iou_loss=False,
                split_iou_loss=True,
                weighted_iou_loss=False,
                gamma=2,
                corner_loss=False,
                weighted_box_loss=False,
                finetune_step=30000,
                part_type="quadrant",
                part_classification=False,
                part_classification_weight=5.0,
                smoothl1_loss_weight=5.0,
                cls_loss_weight=50, #TODO changed classification loss weight from 10 to 50
                use_occupancy=False,
                centerness=None,
                refinement=None,
                range=[-50.4, -50.4, -5.0, 50.4, 50.4, 3.0],
                tasks=tasks,
                vel_branch=True,  # Added velocity branch
                #ToDo make a whole loss sub_dict
                code_weights=[1.0]*6 +[0.2]*2 +[1.0]*2, # [5.0, 5.0, 7.0, 3.0, 3.0, 5.0, 0.0, 0.0, 5.0, 5.0], # This includes velocity in positions -4, -3.
            ),
        ),
    ),

    bbox_head=dict(
            # type='RPNHead',
            type="MultiGroupHeadOHS",
            mode="3d",
            in_channels=sum([128,]),
            norm_cfg=norm_cfg,
            tasks=tasks,
            weights=[1,],
            box_coder=build_box_coder(box_coder),
            encode_background_as_zeros=True,
            loss_norm=dict(
                type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,
            ),
            loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
            use_sigmoid_score=True,
            loss_bbox=dict(
                type="WeightedSmoothL1Loss",
                sigma=3.0,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0],
                codewise=True,
                loss_weight=2.0,
            ),
            encode_rad_error_by_sin=True,
            loss_aux=dict(
                type="WeightedSoftmaxClassificationLoss",
                name="direction_classifier",
                loss_weight=0.2,
            ),
            direction_offset=0.785, # For direction classifier?
        ),
    # # Not used
    # loss=dict(
    #     classification_loss=dict(
    #         weighted_sigmoid_focal=dict(
    #             alpha=0.25,
    #             gamma=2.0,
    #             anchorwise_ouput=True,
    #         ),
    #     ),
    #     localization_loss=dict(
    #         weighted_smooth_l1=dict(
    #         sigma=3.0,
    #         code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #         ),
    #     ),
    #     classification_weight=1.0,
    #     localization_weight=2.0,
    # ),
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
    ohs=True, # Added OHS
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=100,
        nms_iou_threshold=0.01, # 0.1 for Claire's KITTI
    ),
    score_threshold=0.3,
    post_center_limit_range=[0, -40.0, -5.0, 70.4, 40.0, 5.0],
    max_per_img=100,
)

# dataset settings
# dataset_type = "KittiDataset"
# data_root = "data/KITTI"

dataset_type = "NuScenesDataset"
n_sweeps = 10
data_root = "data/Nuscenes/v1.0-trainval"


db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/Nuscenes/v1.0-trainval/dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[dict(car=15,),],
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(car=5,)),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    # mode="train",
    # shuffle_points=True,
    # gt_loc_noise=[1.0, 1.0, 0.5],
    # gt_rot_noise=[-0.785, 0.785],
    # global_rot_noise=[-0.785, 0.785],
    # global_scale_noise=[0.95, 1.05],
    # global_rot_per_obj_range=[0, 0],
    # global_trans_noise=[0.0, 0.0, 0.0],
    # remove_points_after_sample=True,
    # gt_drop_percentage=0.0,
    # gt_drop_max_keep_points=15,
    # remove_unknown_examples=False,
    # remove_environment=False,
    # db_sampler=db_sampler,
    # class_names=class_names,
    # ohs = False,
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.0, 0.0, 0.0],  # Claire's KITTI values [1.0, 1.0, 0.5]
    gt_rot_noise=[0.0, 0.0],  # Claire's KITTI values [-0.3141592654, 0.3141592654]
    global_rot_noise=[-0.3925, 0.3925],  # Claire's KITTI values [-0.78539816, 0.78539816]
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0,0,0], #[0.2, 0.2, 0.2], # [0,0,0] for claire's KITTI
    remove_points_after_sample=False, # True for Claire's
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    # range=[0, -40.0, -3.0, 70.4, 40.0, 1.0],
    # voxel_size=[0.05, 0.05, 0.1],
    # max_points_in_voxel=5,
    # max_voxel_num=20000,
    range=[-50.4, -50.4, -5.0, 50.4, 50.4, 3.0],
    voxel_size=[0.1, 0.1, 0.2],
    max_points_in_voxel=10,
    max_voxel_num=60000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

# train_anno = "data/KITTI/kitti_infos_train.pkl"
# val_anno = "data/KITTI/kitti_infos_val.pkl"
train_anno = "data/Nuscenes/v1.0-trainval/infos_train_10sweeps_repeat_withvelo_resampled.pkl"
val_anno = "data/Nuscenes/v1.0-trainval/infos_val_10sweeps_repeat_withvelo.pkl"
test_anno = None

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_path=train_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        anno_path=test_anno,
        n_sweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False, # 1e-3 for KITTI
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2)) # 10 in Claire's KITTI
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs =24
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "experiments/SECOND"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
