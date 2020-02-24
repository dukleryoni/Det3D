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
