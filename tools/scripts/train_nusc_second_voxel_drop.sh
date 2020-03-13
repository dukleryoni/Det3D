#!/bin/bash
TASK_DESC=$1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
#OUT_DIR=experiments/nusc_second/voxel_drop/gt_drop_02
OUT_DIR=experiments/nusc_second/baselines/ohs_again

#general_voxel_01

NUSC_SECOND=$OUT_DIR
NUSC_CBGS_WORK_DIR=$OUT_DIR
LYFT_CBGS_WORK_DIR=$OUT_DIR/LYFT_CBGS_$TASK_DESC\_$DATE_WITH_TIME
SECOND_WORK_DIR=$OUT_DIR/SECOND_$TASK_DESC\_$DATE_WITH_TIME
PP_WORK_DIR=$OUT_DIR/PointPillars_$TASK_DESC\_$DATE_WITH_TIME

RESUME_FROM=$OUT_DIR/latest.pth

if [ ! $TASK_DESC ]
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

# Voxelnet
# CUDA_VISIBLE_DEVICES=3
# CUDA_LAUNCH_BLOCKING=1
#--resume_from=$RESUME_FROM
#CUDA_VISIBLE_DEVICES=0,1,2,3
#CUDA_VISIBLE_DEVICES=4,5,6,7
#python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/hpc_ohs_nusc_car_vfev3_RESNET_rpn1_mghead_syncbn.py --work_dir=$NUSC_SECOND
python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/hpc_ohs_nusc_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$NUSC_SECOND --resume_from=$RESUME_FROM

#--gt_drop --drop_pct=20
# --resume_from "experiments/nusc_second/latest.pth"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/second/configs/nusc_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py --work_dir=$NUSC_SECOND


#python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_SECOND
# -m torch.distributed.launch --nproc_per_node=4
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$LYFT_CBGS_WORK_DIR

# python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
# PointPillars
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./examples/point_pillars/configs/original_pp_mghead_syncbn_kitti.py --work_dir=$PP_WORK_DIR
# --nproc_per_node=4
# ohs_local_