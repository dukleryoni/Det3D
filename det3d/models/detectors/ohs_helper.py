import time, pdb,os
from enum import Enum
import contextlib
import numpy as np
import torch, cv2
from torch import nn
import gc
from det3d.core import box_torch_ops
from det3d.ops.iou3d.iou3d_utils import boxes_iou3d_gpu


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period

def cat(t):
    if len(t) > 1:
        return torch.cat(t)
    else:
        return t[0]

def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot



def _get_pos_neg_loss(cls_loss, labels):
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or cls_loss.dim == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


# REGISTERED_NETWORK_CLASSES = {}


# def register_voxelnet(cls, name=None):
#     global REGISTERED_NETWORK_CLASSES
#     if name is None:
#         name = cls.__name__
#     assert name not in REGISTERED_NETWORK_CLASSES, f"exist class: {REGISTERED_NETWORK_CLASSES}"
#     REGISTERED_NETWORK_CLASSES[name] = cls
#     return cls
#
#
# def get_voxelnet_class(name):
#     global REGISTERED_NETWORK_CLASSES
#     assert name in REGISTERED_NETWORK_CLASSES, f"available class: {REGISTERED_NETWORK_CLASSES}"
#     return REGISTERED_NETWORK_CLASSES[name]


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"




#### Helper Functions ####
def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot
    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]],
                       dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                       dim=-1)
    return boxes1, boxes2

def iou3d(boxes1, boxes2):
    iou = torch.zeros((0,), device=boxes1.device, dtype=torch.float)
    for box1, box2 in zip(boxes1, boxes2):
        _, iou_single = boxes_iou3d_gpu(box1.view(1, -1), box2.view(1, -1))
        iou = torch.cat([iou, iou_single])
    return iou

def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                iou_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                sin_error_factor=1.0,
                box_code_size=7,
                num_direction_bins=2,
                batch_anchors=None,
                iou_loss_ftor=None):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    if iou_preds is not None:
        mask = reg_weights.view(-1) > 0
        positive_box_preds = (box_preds.clone().detach().view(-1, box_code_size))[mask]
        positive_box_preds[:, 3:6] = torch.exp(positive_box_preds[:, 3:6])
        positive_reg_targets = (reg_targets.clone().detach().view(-1, box_code_size))[mask]
        positive_reg_targets[:, 3:6] = torch.exp(positive_reg_targets[:, 3:6])
        iou_preds = iou_preds.view(-1, 1)[mask]
        batch_anchors = batch_anchors.view(-1, 3)
        batch_anchors = batch_anchors[mask]
        positive_box_preds[:, 3:6]*= batch_anchors
        positive_reg_targets[:, 3:6]*= batch_anchors
        diags = torch.sqrt(batch_anchors[:, 0]*batch_anchors[:, 0]+batch_anchors[:, 1]*batch_anchors[:, 1])
        positive_box_preds[:, 0] *= diags
        positive_reg_targets[:, 0] *= diags
        positive_box_preds[:, 1] *= diags
        positive_reg_targets[:, 1] *= diags
        positive_box_preds[:, 2] *= batch_anchors[:, 2]
        positive_reg_targets[:, 2] *= batch_anchors[:, 2]
        iou_gt = iou3d(positive_box_preds, positive_reg_targets)
        iou_loss=iou_loss_ftor(iou_preds, iou_gt).mean(0)
    else:
        iou_loss = 0
    '''
    shabi_score = torch.sigmoid(cls_preds).data.cpu().numpy()
    shabi_cls = np.argmax(shabi_score, -1)
    shabi_score = shabi_score.reshape([-1, shabi_score.shape[-1]])
    shabi_cls = shabi_cls.reshape(-1)
    mask = (shabi_score[np.arange(len(shabi_score)), shabi_cls] < 0.3)
    shabi_cls += 1
    shabi_cls[mask]=0
    shabi_targets=cls_targets.data.cpu().numpy().reshape(-1)
    '''
    #print('gt ratio', (shabi_targets>0).astype(np.float).sum()/len(shabi_targets))
    #print('cls_acc:', np.sum((shabi_cls[shabi_targets>0]==shabi_targets[shabi_targets>0]).astype(np.float))/shabi_targets.astype(np.float).sum())
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        # reg_tg_rot = box_torch_ops.limit_period(
        #     reg_targets[..., 6:7], 0.5, 2 * np.pi / num_direction_bins)
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets, box_preds[..., 6:7], reg_targets[..., 6:7],
                                                    sin_error_factor)

    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses, iou_loss


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors,
                         reg_targets,
                         one_hot=True,
                         dir_offset=0,
                         num_bins=2):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = box_torch_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = one_hot(
            dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets


def get_direction_target_fsaf(rot_gt,
                         one_hot=True,
                         dir_offset=0,
                         num_bins=2):
    offset_rot = box_torch_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = one_hot(
            dir_cls_targets, num_bins, dtype=torch.float)
    return dir_cls_targets


def _meshgrid(x, y):
    xx = x.repeat(len(y))
    yy = y.contiguous().view(-1, 1).repeat(1, len(x)).contiguous().view(-1)
    return xx, yy

def generate_points(featmap_size, pc_range, device='cuda', dtype=torch.float32):
    feat_h, feat_w = featmap_size
    dims = pc_range[3:5]-pc_range[:2]
    shift_x = (torch.arange(0, feat_w, device=device, dtype=dtype) + 0.5) / feat_w * dims[0] + pc_range[0]
    shift_y = (torch.arange(0, feat_h, device=device, dtype=dtype) + 0.5) / feat_h * dims[1] + pc_range[1]
    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shift_zz = torch.ones(shift_xx.shape, device=device) * 0.5 * (pc_range[2]+pc_range[5])
    points = torch.stack((shift_xx, shift_yy, shift_zz), dim=-1)
    return points

#
# def center_to_corner_box2d(center, dims, cos, sin):
#     boxes_bev = center.new(torch.Size((center.shape[0], 4, 2)))
#     half_l, half_w = dims[:, 0] / 2, dims[:, 1] / 2
#     boxes_bev[:, 0, 0], boxes_bev[:, 0, 1] = -half_l, -half_w
#     boxes_bev[:, 1, 0], boxes_bev[:, 1, 1] = -half_l, half_w
#     boxes_bev[:, 2, 0], boxes_bev[:, 2, 1] = half_l, half_w
#     boxes_bev[:, 3, 0], boxes_bev[:, 3, 1] = half_l, -half_w
#     rot_mat_T = torch.stack(
#           [tstack([cos, -sin]),
#            tstack([sin, cos])])
#     return torch.einsum('aij, jka->aik', (boxes_bev, rot_mat_T)) + center.unsqueeze(1)
#
#
#
# def 3D_points_in_gt_box_torch(points, gt_bboxes, clockwise=True):
#     """check points is in convex polygons. may run 2x faster when write in
#     cython(don't need to calculate all cross-product between edge and point)
#     Args:
#         points: [num_points, 2] array.
#         polygon: [num_polygon, num_points_of_polygon, 2] array.
#         clockwise: bool. indicate polygon is clockwise.
#     Returns:
#         [num_points, num_polygon] bool array.
#     """
#     gt_centers = gt_bboxes[...,:3]
#     gt_dims = gt_bboxes[...,3:6]
#     gt_yaw = gt_bboxes[...,-1]
#     centered_points = points.unsqueeze(1) - gt_centers.unsqueeze(0)
#     cos, sin = torch.cos(gt_yaw), torch.sin(gt_yaw)
#
#     rot_mat_T = torch.stack(
#     [tstack([cos, -sin]),
#     tstack([sin, cos])])
#     centered_points[..., :2] = torch.einsum('aij, jka->aik', (centered_points[..., :2], rot_mat_T))
#     correct_idx = torch.all(torch.abs(centered_points) < gt_dims) # fix the dimension
#     correct_points = hh[correct_idx], ww[correct_idx]
#     data_idx = data in correct_points
#
#     # first convert polygon to directed lines
#     num_lines = polygon.shape[1]
#     polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
#     if clockwise:
#         vec1 = (polygon - polygon_next)
#     else:
#         vec1 = (polygon_next - polygon)
#     vec1 = vec1.unsqueeze(0)
#     vec2 = polygon.unsqueeze(0) - points.unsqueeze(1).unsqueeze(1)
#     vec2 = vec2[..., [1, 0]]
#     vec2[..., 1] *= -1
#     cross = (vec1 * vec2).sum(-1)
#
#     return torch.all(cross > 0, dim=2)

def coor_batch(coors):
    '''

    Args:
        coors: LongTensor describing the list of coordinates of non-empty voxels. Each coordinates is of the form (batch_location,l,w,h)

    Returns:
        List of coordinate tensors seperated by batch
    '''
    unique_elems = torch.unique(coors[:,0])
    masks = [coors[:,0] == elem for elem in unique_elems]
    return [coors[mask] for mask in masks]


def get_gt_masks(data,range):
    '''

    Args:
        data: includes ground truth targets, coordinates, and things like input features shape
        range: the point cloud range

    Returns:
        A mask for the coordinates tensor that selects the voxels that contain the ground truth

    '''
    pc_range = torch.FloatTensor(range)
    voxel_dims = pc_range[3:] - pc_range[:3]
    voxel_dims /= torch.FloatTensor(data["input_shape"])  # fix for specifics

    coors = data["coors"] # will have the first coordinate to donate element of which batch
    voxel_dims = voxel_dims.to(coors.device)
    coors_batch = coor_batch(coors)
    batch_gt_masks = []
    for i, coor in enumerate(coors_batch):
        gt_boxes = data["fsaf_targets"][i][..., :-1]  # Drop class label
        gt_boxes = gt_boxes.squeeze(0)
        gt_centers = gt_boxes[..., :3]
        gt_dims = gt_boxes[..., 3:6]
        gt_yaw = gt_boxes[..., -1]
        zyx_points = coor[...,1:].type(torch.float)
        points = 0.5 + zyx_points[...,[2,1,0]] # Flip into xyz points, center voxels by 0.5
        points = points * voxel_dims + pc_range[:3].to(coor.device)

        vec = points.unsqueeze(1) - gt_centers.unsqueeze(0)

        yaw_mtx = lambda yaw: torch.FloatTensor([[torch.cos(yaw), torch.sin(yaw)], [torch.sin(yaw), -torch.cos(yaw)]]).to(yaw.device)

        yaw_mtx_list = [yaw_mtx(yaw) for yaw in gt_yaw]
        rot_mtxs = torch.stack(yaw_mtx_list, dim=0)

        rotated_pts = torch.einsum('ijk,jlk-> ijl', vec[..., :2], rot_mtxs)
        # print(rot_mtxs.shape, vec.shape, rotated_pts.shape)
        effective = 1.5 # TODO Can also add an effective gt_box parameter
        vec[..., :2] = rotated_pts
        mask_all_gt = (abs(vec) < effective * (gt_dims + 2*voxel_dims)/2.0).all(dim=-1)  # Added gt_dims + voxel_dims to include any voxel that has an overlap with the box
        mask_all_gt = torch.transpose(mask_all_gt, -2, -1) # TODO Understand why this results in empty gt voxels sometimes, possible: rotation is opposite
        batch_gt_masks.append(mask_all_gt)
        #coors_gt_list = [coors[mask] for mask in mask_all_gt]

    return batch_gt_masks


def get_gt_dropout(drop_rate, features, coors, masks):
    batch_coors = coor_batch(coors)
    batch_gt_masks = masks

    all_masks = []
    for j, coor in enumerate(batch_coors):
        dropout_mask = torch.bernoulli(drop_rate * torch.ones(len(coor))).type(torch.bool).to(coor.device)
        coor_mask = batch_gt_masks[j][0]
        for mask in batch_gt_masks[j]:
            coor_mask+= mask
        coor_mask = coor_mask*dropout_mask
        # print(len(coor_mask), sum(coor_mask.long()))
        keep_mask = ~ coor_mask
        all_masks.append(keep_mask)

    device = all_masks[0].device
    all_masks = [mask.to(device) for mask in all_masks]
    gt_dropout_mask = torch.cat(all_masks)
    return gt_dropout_mask



