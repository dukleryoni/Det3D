import torch, torch.nn as nn, torch.nn.functional as F
import pdb
import numpy as np, sys
from det3d.core.bbox.geometry import points_in_convex_polygon_torch
from det3d.models.losses import WeightedSmoothL1Loss, WeightedL2LocalizationLoss, SigmoidFocalLoss
from functools import partial
from det3d.ops.iou3d.iou3d_utils import boxes_iou3d_gpu
import time
from torch import stack as tstack


def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot

distance = lambda p1, p2, p3 : np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def gaussian_torch(x, mu, sig):
    return torch.exp(-((x - mu) / sig )**2 / 2) / ((2*np.pi) ** 0.5 * sig)

def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return(x, y)

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

# def weighted_sigmoid_focal_loss(pred,
#                                 target,
#                                 weight,
#                                 gamma=2.0,
#                                 alpha=0.25,
#                                 avg_factor=None,
#                                 num_classes=80):
#     if avg_factor is None:
#         avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
#     #print(avg_factor, weight, sigmoid_focal_loss(pred, target, gamma, alpha, 'none'))
#     return torch.sum(
#         sigmoid_focal_loss(pred, target, gamma, alpha, 'none') * weight.view(# bbox_weights过滤了不计算的loss
#             -1, 1))[None] / avg_factor

def center_to_corner_box2d(center, dims, cos, sin):
    boxes_bev = center.new(torch.Size((center.shape[0], 4, 2)))
    half_l, half_w = dims[:, 0] / 2, dims[:, 1] / 2
    boxes_bev[:, 0, 0], boxes_bev[:, 0, 1] = -half_l, -half_w
    boxes_bev[:, 1, 0], boxes_bev[:, 1, 1] = -half_l, half_w
    boxes_bev[:, 2, 0], boxes_bev[:, 2, 1] = half_l, half_w
    boxes_bev[:, 3, 0], boxes_bev[:, 3, 1] = half_l, -half_w
    rot_mat_T = torch.stack(
          [tstack([cos, -sin]),
           tstack([sin, cos])])
    return torch.einsum('aij, jka->aik', (boxes_bev, rot_mat_T)) + center.unsqueeze(1)

def corner_loss_2d(pred_tensor, target_tensor, weight, loss_func, avg_factor=None):
    pred = pred_tensor.clone()
    target = target_tensor.clone().detach()
    pred[:, 3:6] = torch.exp(pred[:, 3:6])
    target[:, 3:6] = torch.exp(target[:, 3:6])
    assert pred.size(0) == target.size(0), "size 0 does not match"
    assert pred.size(1) == target.size(1), "size 1 does not match"
    if avg_factor is None:
       avg_factor = pred.size(0)
    pred_corners = center_to_corner_box2d(pred[:, :2], pred[:, 3:5], pred[:, 6], pred[:, 7])
    target_corners = center_to_corner_box2d(target[:, :2], target[:, 3:5], target[:, 6], target[:, 7])
    corner_losses = loss_func(pred_corners, target_corners,  weights=torch.ones(pred_corners.shape[:-1], device=pred.device))
    return torch.sum(weight * corner_losses)/ avg_factor


class RefineMultiBoxFSAFLoss(nn.Module):
    """FSAF Loss Function
    """

    def __init__(self, cfg, num_classes, pc_range, encode_background_as_zeros=True, use_iou_branch=False):
        super(RefineMultiBoxFSAFLoss, self).__init__()
        # self.input_shape = input_shape
        self.cfg = cfg
        self.cls_out_channels = num_classes-1
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.dims = self.pc_range[3:]-self.pc_range[:3]
        if self.cfg.rot_type == 'cos_sin':
            self.loss_box = WeightedSmoothL1Loss(3.0, [5.0, 5.0, 7.0, 3.0, 3.0, 5.0, 5.0, 5.0])
        if self.cfg.rot_type == 'softbin':
            self.loss_box = WeightedSmoothL1Loss(3.0, [5.0, 5.0, 5.0, 3.0, 3.0, 5.0, 5.0])
        if self.cfg.rot_type == 'softbin_cos_sin':
            self.loss_box = WeightedSmoothL1Loss(3.0, [5.0, 5.0, 5.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0])
        self.loss_ang_vector = WeightedL2LocalizationLoss(5.0)

        self.loss_cls = SigmoidFocalLoss(2.0, 0.25) # hard coded
        if self.cfg.centerness:
            self.loss_centerness = WeightedSmoothL1Loss(self.cfg.centerness_weight, [1.0, 1.0])
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_iou_branch = use_iou_branch
        if use_iou_branch:
            self.loss_iou = SigmoidFocalLoss(2.0, 0.25)
        self.step = 0

    def iou3d(self, pred, target, mask=False):
        sample_pred = pred.clone().detach()
        sample_target = target.clone().detach()
        sample_pred[:, 3:6] = torch.exp(sample_pred[:, 3:6])
        sample_target[:, 3:6] = torch.exp(sample_target[:, 3:6])
        if self.cfg.rot_type == 'cos_sin':
            sample_pred[:, 6] = torch.atan2(sample_pred[:, 7], sample_pred[:, 6])
            sample_target[:, 6] = torch.atan2(sample_target[:, 7], sample_target[:, 7])
            sample_pred = sample_pred[:, :7]
            sample_target = sample_target[:, :7]
        iou_bev = torch.zeros((0,), device=pred.device, dtype=torch.float)
        iou_3d = torch.zeros((0,), device=pred.device, dtype=torch.float)
        for sp, st in zip(sample_pred, sample_target):
            iou_bev_single, iou_3d_single = boxes_iou3d_gpu(sp.reshape(1, -1), st.reshape(1, -1))
            iou_bev = torch.cat([iou_bev, iou_bev_single])
            iou_3d = torch.cat([iou_3d, iou_3d_single])
        if mask:
            area = sample_target[:, 3] * sample_target[:, 4]
            mask = ((area >= 4) & (iou_3d.reshape(-1) >= 0.4)) | ((area < 4) & (iou_3d.reshape(-1) >= 0.25))
            return iou_3d, iou_bev, mask
        '''
        print('==========================')
        print('true_bev_iou:', iou_bev[0].cpu().numpy().reshape(-1), 'true_iou:', iou_3d[0].cpu().numpy().reshape(-1))
        print('==========================')
        '''
        return iou_3d, iou_bev

    def select_iou_loss(self, pred_tensor, target_tensor, weight, avg_factor=None):
        pred = pred_tensor.clone()
        target = target_tensor.clone().detach()
        if self.cfg.dim_type == 'log':
           pred[:, 3:6] = torch.exp(pred[:, 3:6])
           target[:, 3:6] = torch.exp(target[:, 3:6])
        if avg_factor is None:
            avg_factor = pred.size(0)
        assert pred.size(0) == target.size(0), "size 0 does not match"
        assert pred.size(1) == target.size(1), "size 1 does not match"
        if self.cfg.dim_type == 'distance':
            area_pred = (pred[:, 0] + pred[:, 3]) * (pred[:, 1] + pred[:, 4])
            area_gt = (target[:, 0] + target[:, 3]) * (target[:, 1] + target[:, 4])
            area_i = ((torch.min(pred[:, 0], target[:, 0]) +
                       torch.min(pred[:, 3], target[:, 3])) *
                       (torch.min(pred[:, 1], target[:, 1]) +
                       torch.min(pred[:, 4], target[:, 4])))
            h_pred = pred[:, 2] + pred[:,5]
            h_gt = target[:, 2] + target[:, 5]
            h_i = torch.min(pred[:, 2], target[:, 2]) + torch.min(pred[:, 5], target[:, 5])
        else:
            area_pred = pred[:, 3] * pred[:, 4]
            area_gt = target[:, 3] * target[:, 4]
            if 'center' in self.cfg.loc_type:
                area_i = (torch.min(pred[:, 0] + 0.5 * pred[:, 3], target[:, 0] + 0.5 * target[:, 3]) - \
                          torch.max(pred[:, 0] - 0.5 * pred[:, 3], target[:, 0] - 0.5 * target[:, 3])) * \
                          (torch.min(pred[:, 1] + 0.5 * pred[:, 4], target[:, 1] + 0.5 * target[:, 4]) - \
                          torch.max(pred[:, 1] - 0.5 * pred[:, 4], target[:, 1] - 0.5 * target[:, 4]))
                area_c = (torch.max(pred[:, 0] + 0.5 * pred[:, 3], target[:, 0] + 0.5 * target[:, 3]) - \
                          torch.min(pred[:, 0] - 0.5 * pred[:, 3], target[:, 0] - 0.5 * target[:, 3])) * \
                          (torch.max(pred[:, 1] + 0.5 * pred[:, 4], target[:, 1] + 0.5 * target[:, 4]) - \
                          torch.min(pred[:, 1] - 0.5 * pred[:, 4], target[:, 1] - 0.5 * target[:, 4]))
            if self.cfg.loc_type == 'part':
                area_i = (torch.min(pred[:, 0]*pred[:, 3], target[:, 0]*target[:, 3])+ \
                          torch.min((1-pred[:, 0])*pred[:, 3], (1-target[:, 0])*target[:, 3])) * \
                          (torch.min(pred[:, 1]*pred[:, 4], target[:, 1]*target[:, 4])+ \
                          torch.min((1-pred[:, 1])*pred[:, 4], (1-target[:, 1])*pred[:, 4]))
            h_pred = pred[:, 5]
            h_gt = target[:, 5]
            h_i = torch.min(pred[:, 2] + pred[:, 5]/2, target[:, 2] + target[:, 5]/2) - \
                  torch.max(pred[:, 2] - pred[:, 5]/2, target[:, 2] - target[:, 5]/2)
        area_u = area_pred + area_gt - area_i
        v_pred = h_pred * area_pred
        v_gt = h_gt * area_gt
        v_i = area_i * h_i
        v_u = v_pred + v_gt - v_i
        #iou = area_i / area_u * torch.abs(torch.cos(target[:, -1]-pred[:, -1]))
        if self.cfg.rot_type == 'cos_sin':
            iou = v_i / v_u * torch.abs(target[:, 6]*pred[:, 6]+target[:, 7]*pred[:, 7])
        else:
            iou = v_i / v_u * torch.abs(torch.cos(target[:, -1]-pred[:, -1]))
        if torch.sum(pred!=pred)>0 or torch.sum(area_u == 0)>0:
            print('iou:', iou, area_pred, area_gt, area_i, '\n')
            import sys
            sys.exit()
        '''
        print('==========================')
        print('estimated_iou', iou[0].detach().cpu(), 'height_iou', (h_i / (h_pred + h_gt - h_i))[0].detach().cpu(), 'pred', pred_tensor[0].detach().cpu(), 'target', target_tensor[0].detach().cpu())
        print('==========================')
        '''
        #weight = (1.8 - iou_3d) ** 5
        if self.cfg.split_iou_loss:
            loc_losses = torch.log(2 - area_i / area_u * torch.abs(target[:, 6]*pred[:, 6]+target[:, 7]*pred[:, 7]) - h_i / (h_pred + h_gt - h_i))
        else:
            loc_losses = torch.log(1 - iou)
        if self.cfg.weighted_iou_loss:
            if self.step >= self.cfg.finetune_step:
                _, iou_bev = self.iou3d(pred_tensor, target_tensor)
                weight = (1.8 - iou_bev) ** self.cfg.gamma
        return torch.sum(weight * loc_losses) / avg_factor

    def select_iou_loss_part_regression(self, pred_tensor, target_tensor, weight, avg_factor=None):
        pred = pred_tensor.clone()
        target = target_tensor.clone().detach()
        pred[:, 3:6] = torch.exp(pred[:, 3:6])
        target[:, 3:6] = torch.exp(target[:, 3:6])
        #pred[:, 2] *= 2
        #pred[:, 2] += 0.5 * (self.pc_range[2]+self.pc_range[5])
        #target[:, 2] = target[:, 2] * 2 + 0.5 * (self.pc_range[2]+self.pc_range[5])
        if avg_factor is None:
            avg_factor = pred.size(0)
        assert pred.size(0) == target.size(0), "size 0 does not match"
        assert pred.size(1) == target.size(1), "size 1 does not match"
        area_pred = pred[:, 3] * pred[:, 4]
        area_gt = target[:, 3] * target[:, 4]
        area_i = (torch.min(pred[:, 0]*pred[:, 3], target[:, 0]*target[:, 3])+ \
                  torch.min((1-pred[:, 0])*pred[:, 3], (1-target[:, 0])*target[:, 3])) * \
                  (torch.min(pred[:, 1]*pred[:, 4], target[:, 1]*target[:, 4])+ \
                   torch.min((1-pred[:, 1])*pred[:, 4], (1-target[:, 1])*pred[:, 4]))
        area_u = area_pred + area_gt - area_i
        h_pred = pred[:, 5]
        h_gt = target[:, 5]
        h_i = torch.min(pred[:, 2] + pred[:, 5]/2, target[:, 2] + target[:, 5]/2) - \
              torch.max(pred[:, 2] - pred[:, 5]/2, target[:, 2] - target[:, 5]/2)
        v_pred = h_pred * area_pred
        v_gt = h_gt * area_gt
        v_i = area_i * h_i
        v_u = v_pred + v_gt - v_i
        #iou = area_i / area_u * torch.abs(torch.cos(target[:, -1]-pred[:, -1]))
        if self.cfg.rot_type == 'cos_sin':
            iou = v_i / v_u * torch.abs(target[:, 6]*pred[:, 6]+target[:, 7]*pred[:, 7])
        else:
            iou = v_i / v_u * torch.abs(torch.cos(target[:, -1]-pred[:, -1]))
        if torch.sum(pred!=pred)>0 or torch.sum(area_u == 0)>0:
            print('iou:', iou, area_pred, area_gt, area_i, '\n')
            import sys
            sys.exit()
        #loc_losses = 1 - iou
        loc_losses = 2 - area_i / area_u * torch.abs(target[:, 6]*pred[:, 6]+target[:, 7]*pred[:, 7]) - h_i / (h_pred + h_gt - h_i)
        sample_pred = pred[0].clone().detach().reshape(1, -1)
        sample_target = target[0].clone().detach().reshape(1, -1)
        mat_rot_pred = torch.tensor([[sample_pred[6], -sample_pred[7]],
                                    [sample_pred[7], sample_pred[6]]], device=pred.device)
        mat_rot_gt = torch.tensor([[sample_target[6], -sample_target[7]],
                                    [sample_target[7], sample_target[6]]], device=pred.device)
        centers = ((sample_pred[:, :2]-0.5) * sample_pred[:, 2:4]).reshape(1,2)
        sample_pred[:, :2] = -centers @ mat_rot_pred
        centers = ((sample_target[:, :2]-0.5) * sample_target[:, 2:4]).reshape(1,2)
        sample_target[:, :2] = -centers @ mat_rot_gt
        sample_pred[:, 6] = torch.atan2(sample_pred[:, 7], sample_pred[:, 6])
        sample_target[:, 6] = torch.atan2(sample_target[:, 7], sample_target[:, 7])
        iou_bev, true_iou = boxes_iou3d_gpu(sample_pred[:, :7], sample_target[:, :7])
        print('==========================')
        try:
            print('true_bev_iou:', iou_bev.cpu().numpy().reshape(-1), 'true_iou:', true_iou.cpu().numpy().reshape(-1), 'estimated_iou', iou[0].detach().cpu(), 'height_iou', (h_i / (h_pred + h_gt - h_i))[0].detach().cpu(), 'pred', pred_tensor[0].detach().cpu(), 'target', target_tensor[0].detach().cpu())
        except:
            print('shabi')
        print('==========================')
        return torch.sum(weight * loc_losses) / avg_factor, true_iou

    def forward(self, fsaf_data, targets, occupancy=None):
        """
        FSAF Loss
        Args:
        """
        gt_bboxes = []
        gt_labels = []
        img_num = len(targets)
        for idx in range(img_num):
            gt_bboxes.append(targets[idx][0,:, :-1])
            gt_labels.append(targets[idx][0,:, -1])
            # gt_bboxes.append(targets[idx][:, :-1].data)
            # gt_labels.append(targets[idx][:, -1].data.long())
        input_loss = fsaf_data + (gt_bboxes, gt_labels)
        fsaf_loss = self.loss(*input_loss, occupancy, gt_bboxes_ignore=None)
        self.step += 1
        return fsaf_loss

    def loss(self, cls_scores, bbox_preds, iou_preds, part_preds, features, gt_bboxes, gt_labels, occupancy, gt_bboxes_ignore=None):
        cls_reg_targets = self.point_target(cls_scores, bbox_preds, gt_bboxes, occupancy, gt_labels_list=gt_labels, gt_bboxes_ignore_list=gt_bboxes_ignore)
        labels_list, label_weights_list, bbox_targets_list, bbox_locs_list, num_total_pos, num_total_neg, iou_weights_list, part_labels_list = cls_reg_targets
        num_total_samples = num_total_pos
        losses_cls, losses_reg = multi_apply(self.loss_single, cls_scores, bbox_preds, iou_preds, part_preds, labels_list, label_weights_list, bbox_targets_list, bbox_locs_list, iou_weights_list, part_labels_list, num_total_samples=num_total_samples)
        ret_refinement = None
        return losses_cls, losses_reg, cls_scores, labels_list, ret_refinement

    def loss_single(self, cls_score, bbox_pred, iou_pred, part_preds, labels, label_weights, bbox_targets, bbox_locs, iou_weights, part_labels, num_total_samples):
        bs_per_device = labels.shape[0]
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        one_hot_targets = one_hot(labels, depth=self.cls_out_channels+1, dtype=cls_score.dtype)
        if self.encode_background_as_zeros:
            one_hot_targets = one_hot_targets[..., 1:]
        loss_cls = self.cfg.cls_loss_weight * self.loss_cls(cls_score.unsqueeze(0), one_hot_targets.unsqueeze(0), weights=label_weights.unsqueeze(0)).sum() / cls_score.numel()    # , ignore_nan_targets=False
        loss_box = bbox_pred.new_zeros(1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        bbox_pred = bbox_pred[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :]

        if self.use_iou_branch:
            iou_pred = iou_pred.permute(0, 2, 3, 1)
            iou_pred = iou_pred[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :]

        if self.cfg.part_classification:
            part_preds = part_preds.permute(0, 2, 3, 1)
            part_preds = part_preds[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :]
            loss_box += self.cfg.part_classification_weight * F.binary_cross_entropy(part_preds, part_labels.squeeze(0))

        if self.cfg.centerness:
            part_preds = part_preds.permute(0, 2, 3, 1)
            part_preds = part_preds[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :]
            loss_box += self.loss_centerness(part_preds, part_labels,  weights=torch.ones(len(bbox_pred), device=bbox_pred.device)).sum()/bbox_pred.size(0)  # ignore_nan_targets=False,

        if self.cfg.iou_loss:
            loss_box += self.select_iou_loss(bbox_pred, bbox_targets, 1.0, avg_factor=num_total_samples)

        if self.cfg.corner_loss:
            if self.step >= self.cfg.finetune_step:
                loss_box += 0.5 * corner_loss_2d(bbox_pred, bbox_targets, 1.0, self.loss_ang_vector, avg_factor=num_total_samples)
        '''
        if not 'cos_sin' in self.cfg.rot_type:
            loss_box += self.loss_ang_vector(torch.cos(bbox_pred[..., 6:7]), torch.cos(bbox_targets[..., 6:7]),  weights=torch.ones(len(bbox_pred), device=bbox_pred.device)).sum() / bbox_pred.size(0)
            loss_box += self.loss_ang_vector(torch.sin(bbox_pred[..., 6:7]), torch.sin(bbox_targets[..., 6:7]),  weights=torch.ones(len(bbox_pred), device=bbox_pred.device)).sum() / bbox_pred.size(0)
        if self.encode_rad_error_by_sin:
            bbox_pred, bbox_targets = add_sin_difference(bbox_pred, bbox_targets, bbox_pred[..., 6:7], bbox_targets[..., 6:7])
        '''
        if self.cfg.weighted_box_loss and self.step >= self.cfg.finetune_step: # can remove
            true_iou, _ = self.iou3d(bbox_pred, bbox_targets)
            loss_box += self.loss_box(bbox_pred, bbox_targets,  weights=(0.8*torch.exp(1-true_iou)).reshape(-1)).sum()/bbox_pred.size(0)
            '''
            vol = bbox_targets[:, 3] * bbox_targets[:, 4] * bbox_targets[:, 5]
            loss_box += self.loss_box(bbox_pred, bbox_targets,  weights=(2 * torch.exp(-torch.sqrt(vol))).reshape(-1)).sum()/bbox_pred.size(0)
            '''
        else:
            loss_box += self.cfg.smoothl1_loss_weight *  self.loss_box(bbox_pred, bbox_targets,  weights=torch.ones(len(bbox_pred), device=bbox_pred.device)).sum()/bbox_pred.size(0)
        if self.use_iou_branch:
            true_iou, _ = self.iou3d(bbox_pred, bbox_targets)
            '''
            iou_gt[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :] = true_iou
            iou_gt = iou_gt.view(iou_gt.shape[0], -1, iou_gt.shape[-1])
            tmp_iou_pred = iou_pred[bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], :]
            iou_pred = iou_pred.view(iou_pred.shape[0], -1, iou_pred.shape[-1])
            loss_box += 100 * self.loss_iou(iou_pred, iou_gt,  weights=iou_weights.view(iou_weights.shape[0], -1)).sum()/bbox_pred.size(0)
            '''
            loss_box += self.cfg.iou_loss_weight * self.loss_box(iou_pred.unsqueeze(0), true_iou.unsqueeze(0),  weights=iou_pred.new_ones(len(iou_pred))).sum()/bbox_pred.size(0)
            print('====================')
            print('true_iou', true_iou[0].cpu().numpy(), 'iou_pred', iou_pred[0].detach().cpu().numpy())
            print('====================')
        return (loss_cls, loss_box)

    def point_target(self, cls_scores, bbox_preds, gt_bboxes, occupancy=None, gt_labels_list=None, gt_bboxes_ignore_list=None):
        num_imgs = len(gt_bboxes)
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [ None for _ in range(num_imgs)]

        if gt_labels_list is None:
            gt_labels_list = [ None for _ in range(num_imgs)]
        num_levels = len(self.cfg.feat_strides)
        assert len(cls_scores) == len(bbox_preds) == num_levels
        cls_score_list = []
        bbox_pred_list = []
        for img_id in range(num_imgs):
            cls_score_list.append([ cls_scores[i][img_id].detach() for i in range(num_levels) ])
            bbox_pred_list.append([ bbox_preds[i][img_id].detach() for i in range(num_levels) ])
        if occupancy is None: occupancy=[None]*num_imgs

        if 'center' in self.cfg.loc_type:
            all_labels, all_label_weights, all_bbox_targets, all_bbox_locs, num_pos_list, num_neg_list, iou_weights, part_labels = multi_apply(self.point_target_single_center_regression, cls_score_list, bbox_pred_list, gt_bboxes, gt_bboxes_ignore_list, gt_labels_list, occupancy)
        else:
            all_labels, all_label_weights, all_bbox_targets, all_bbox_locs, num_pos_list, num_neg_list = multi_apply(self.point_target_single, cls_score_list, bbox_pred_list, gt_bboxes, gt_bboxes_ignore_list, gt_labels_list)
        for i in range(num_imgs):
            for lvl in range(num_levels):
                all_bbox_locs[i][lvl][:, 0] = i

        num_total_pos = sum([ max(num, 1) for num in num_pos_list ])
        num_total_neg = sum([ max(num, 1) for num in num_neg_list ])
        labels_list = self.images_to_levels(all_labels, num_imgs, num_levels, True)
        label_weights_list = self.images_to_levels(all_label_weights, num_imgs, num_levels, True)
        bbox_targets_list = self.images_to_levels(all_bbox_targets, num_imgs, num_levels, False)
        bbox_locs_list = self.images_to_levels(all_bbox_locs, num_imgs, num_levels, False)
        if self.use_iou_branch:
            iou_weights_list = self.images_to_levels(iou_weights, num_imgs, num_levels, True)
        else:
            iou_weights_list = [None]*len(labels_list)
        if self.cfg.part_classification or self.cfg.centerness:
            part_labels = self.images_to_levels(part_labels, num_imgs, num_levels, False)
        else:
            part_labels = [None]*len(labels_list)
        return (
         labels_list, label_weights_list, bbox_targets_list,
         bbox_locs_list, num_total_pos, num_total_neg, iou_weights_list, part_labels)

    def point_target_single_center_regression(self, cls_score_list, bbox_pred_list, gt_bboxes, gt_bboxes_ignore, gt_labels, occupancy):
        num_levels = len(self.cfg.feat_strides)
        assert len(cls_score_list) == len(bbox_pred_list) == num_levels
        device = bbox_pred_list[0].device
        if num_levels == 1:
            feat_lvls = torch.tensor([0]*gt_bboxes.size(0), dtype=torch.int, device=device)
        else:
            feat_lvls = self.feat_level_select(cls_score_list, bbox_pred_list, gt_bboxes, gt_labels)

        labels = []
        label_weights = []
        bbox_targets = []
        bbox_locs = []

        if self.use_iou_branch:
            gt_ious = []
            gt_iou_weights = []
        if self.cfg.part_classification or self.cfg.centerness:
            part_labels = []
        #img_h, img_w = self.input_shape
        for lvl in range(num_levels):
            stride = self.cfg.feat_strides[lvl]
            norm = self.cfg.norm_factor
            inds = torch.nonzero(feat_lvls == lvl).squeeze(-1)
            h, w = cls_score_list[lvl].size()[-2:]
            valid_h = h
            valid_w = w
            _labels = torch.zeros_like(cls_score_list[lvl][0], dtype=torch.long)
            _label_weights = torch.zeros_like(cls_score_list[lvl][0], dtype=torch.float)
            _label_weights[:valid_h, :valid_w] = 1.
            _bbox_targets = bbox_pred_list[lvl].new_zeros((0, bbox_pred_list[0].size(0)), dtype=torch.float)
            _bbox_locs = bbox_pred_list[lvl].new_zeros((0, 3), dtype=torch.long)
            if self.use_iou_branch:
                _gt_iou_weights = torch.zeros_like(cls_score_list[lvl][0], dtype=torch.float)
                _gt_iou_weights[:valid_h, :valid_w] = 1.
            if self.cfg.part_classification:
                if self.cfg.part_type in ['up_down', 'left_right']:
                    part_num = 2
                elif self.cfg.part_type == 'quadrant':
                    part_num = 4
                else:
                    part_num = 8
                _part_labels = bbox_pred_list[lvl].new_zeros((0, part_num), dtype=torch.float)
            if self.cfg.centerness:
                _part_labels = bbox_pred_list[lvl].new_zeros((0, 2), dtype=torch.float)
            if len(inds) > 0:
                boxes = gt_bboxes[inds, :]
                classes = gt_labels[inds]
                if self.cfg.use_border:
                    ignore_boxes_in = boxes[:, [0,1,3,4]].clone().detach()
                    ignore_boxes_in[:, 2:4] *= self.cfg.s1
                    ignore_boxes_in = center_to_corner_box2d(ignore_boxes_in[:, :2], ignore_boxes_in[:, 2:4], torch.cos(boxes[:,-1]), torch.sin(boxes[:,-1]))
                    ignore_boxes_in[:, :, 0] = (ignore_boxes_in[:, :, 0] - self.pc_range[0]) / self.dims[0] * w
                    ignore_boxes_in[:, :, 1] = (ignore_boxes_in[:, :, 1] - self.pc_range[1]) / self.dims[1] * h

                effective_boxes = boxes[:, [0,1,3,4]].clone().detach()
                effective_boxes[:, 2:4] *= self.cfg.s2
                effective_boxes = center_to_corner_box2d(effective_boxes[:, :2], effective_boxes[:, 2:4], torch.cos(boxes[:,-1]), torch.sin(boxes[:,-1]))
                effective_boxes[:, :, 0] = (effective_boxes[:, :, 0] - self.pc_range[0]) / self.dims[0] * w
                effective_boxes[:, :, 1] = (effective_boxes[:, :, 1] - self.pc_range[1]) / self.dims[1] * h

                ignore_boxes_out = boxes[:, [0,1,3,4]].clone().detach()
                ignore_boxes_out[:, 2:4] *= self.cfg.s3
                ignore_boxes_out = center_to_corner_box2d(ignore_boxes_out[:, :2], ignore_boxes_out[:, 2:4], torch.cos(boxes[:,-1]), torch.sin(boxes[:,-1]))
                ignore_boxes_out[:, :, 0] = (ignore_boxes_out[:, :, 0] - self.pc_range[0]) / self.dims[0] * w
                ignore_boxes_out[:, :, 1] = (ignore_boxes_out[:, :, 1] - self.pc_range[1]) / self.dims[1] * h

                if self.use_iou_branch:
                    ignore_boxes_iou = boxes[:, [0,1,3,4]].clone().detach()
                    ignore_boxes_iou[:, 2:4] *= 2.0
                    ignore_boxes_iou = center_to_corner_box2d(ignore_boxes_iou[:, :2], ignore_boxes_iou[:, 2:4], torch.cos(boxes[:,-1]), torch.sin(boxes[:,-1]))
                    ignore_boxes_iou[:, :, 0] = (ignore_boxes_iou[:, :, 0] - self.pc_range[0]) / self.dims[0] * w
                    ignore_boxes_iou[:, :, 1] = (ignore_boxes_iou[:, :, 1] - self.pc_range[1]) / self.dims[1] * h

                ww, hh = np.meshgrid(range(w), range(h))
                ww = ww.reshape(-1)
                hh = hh.reshape(-1)

                ww = torch.FloatTensor(ww).to(boxes.device)
                hh = torch.FloatTensor(hh).to(boxes.device)

                ww_l = ww.long()
                hh_l = hh.long()

                if self.cfg.part_classification:
                    if self.cfg.part_type in ['up_down', 'left_right']:
                        part_num = 2
                    elif self.cfg.part_type == 'quadrant':
                        part_num = 4
                    else:
                        part_num = 8
                    _part_labels = bbox_pred_list[lvl].new_zeros((0, part_num), dtype=torch.float)

                if self.cfg.centerness:
                    _part_labels = bbox_pred_list[lvl].new_zeros((0, 2), dtype=torch.float)

                for i in range(len(inds)):
                    locs_x = []
                    locs_y = []

                    pos_mask = points_in_convex_polygon_torch(torch.stack([ww+0.5, hh+0.5], 1), effective_boxes[i].unsqueeze(0))

                    if self.cfg.use_border:
                        print("border")
                        pos_mask = pos_mask & (~points_in_convex_polygon_torch(torch.stack([ww+0.5, hh+0.5], 1),
                                                                               ignore_boxes_in[i].unsqueeze(0)))

                    pos_ind = pos_mask.nonzero()[:, 0]
                    # print(pos_mask.sum(), pos_ind)
                    if len(pos_ind)==0:
                        #print(classes[i], boxes[i], effective_boxes[i])
                        continue
                    #assert len(pos_ind)>0, 'no gt in image'

                    if self.cfg.limit_points:
                        if len(pos_ind) > int(self.cfg.num_points/torch.sqrt(boxes[i, 3]*boxes[i, 4])) :
                            points = torch.stack([(ww[pos_ind]+0.5)/w*self.dims[0]+self.pc_range[0], (hh[pos_ind]+0.5)/h*self.dims[1]+self.pc_range[1]], 1)
                            diff = boxes[i, :2]-points
                            diff = torch.norm(diff, dim=1)
                            sorted_ind = torch.argsort(diff)[:self.cfg.num_points]
                            pos_ind = pos_ind[sorted_ind]
                        pos_mask[:] = 0
                        pos_mask[pos_ind]=1

                    if occupancy is not None:
                        # print(pos_mask.is_cuda, occupancy.is_cuda)
                        occupancy = occupancy.view(pos_mask.shape).type(pos_mask.dtype)
                        pos_mask &= occupancy
                    pos_ind = pos_mask.nonzero()[:, 0]
                    pos_hh, pos_ww = hh_l[pos_ind], ww_l[pos_ind]
                    _labels[pos_hh, pos_ww] = classes[i]
                    _label_weights[pos_hh, pos_ww] = 1.0
                    locs_x.append(pos_ww)
                    locs_y.append(pos_hh)


                    ig_mask = points_in_convex_polygon_torch(torch.stack([ww+0.5, hh+0.5], 1), ignore_boxes_out[i].unsqueeze(0))

                    ig_mask = ig_mask & (~pos_mask)
                    ig_ind = ig_mask.nonzero()[:, 0]
                    ig_h, ig_w = hh_l[ig_ind], ww_l[ig_ind]
                    _label_weights[ig_h, ig_w] = 0

                    locs_x = torch.cat(locs_x, 0).view(-1)
                    locs_y = torch.cat(locs_y, 0).view(-1)

                    shift_xx = locs_x.float().reshape(-1) + 0.5
                    shift_yy = locs_y.float().reshape(-1) + 0.5
                    shifts=torch.zeros(shift_xx.shape+(bbox_pred_list[0].size(0),), device=device, dtype=torch.float)
                    centers = torch.stack((shift_xx/w*self.dims[0]+self.pc_range[0], shift_yy/h*self.dims[1]+self.pc_range[1]), -1)

                    if self.cfg.loc_type == 'part':
                        mat_rot_t = torch.tensor([[torch.cos(boxes[i, -1]), torch.sin(boxes[i, -1])],
                                            [-torch.sin(boxes[i, -1]), torch.cos(boxes[i, -1])]], device=device)
                        shifts[:, 0:2] = (centers - boxes[i, 0:2]) @ mat_rot_t
                        shifts[:, 0] = shifts[:, 0]/boxes[i, 3] + 0.5
                        shifts[:, 1] = shifts[:, 1]/boxes[i, 4] + 0.5
                        shifts[:, 0:2] = shifts[:, 0:2].clamp(min=0.0, max=1.0)
                    if 'center' in self.cfg.loc_type:
                        shifts[:, 0:2] = boxes[i, 0:2] - centers
                    if 'bottom' in self.cfg.h_loc_type:
                        shifts[:, 2] = boxes[i, 2] - 0.5 * boxes[i, 5]
                    else:
                        shifts[:, 2] = boxes[i, 2]
                    #shifts[:, 2] = (boxes[i, 2] - self.pc_range[2])/self.dims[2]
                    #shifts[:, 2] = (boxes[i, 2] - 0.5 * (self.pc_range[2]+self.pc_range[5]))/2
                    if self.cfg.rot_type == 'cos_sin':
                        shifts[:, 6] = torch.cos(boxes[i, -1])
                        shifts[:, 7] = torch.sin(boxes[i, -1])
                    else:
                        shifts[:, 6:] = boxes[i, 6:]
                    if self.cfg.rot_type == 'softbin_cos_sin':
                        shifts[:, 7] = torch.cos(boxes[i, -1])
                        shifts[:, 8] = torch.sin(boxes[i, -1])
                    if self.cfg.dim_type == 'log':
                        shifts[:, 3:6] = torch.log(boxes[i, 3:6])
                    else:
                        shifts[:, 3:6] = boxes[i, 3:6]
                    if self.cfg.part_classification:
                        mat_rot_t = torch.tensor([[torch.cos(boxes[i, -1]), torch.sin(boxes[i, -1])],
                                             [-torch.sin(boxes[i, -1]), torch.cos(boxes[i, -1])]], device=device)
                        relative_shifts = (centers - boxes[i, 0:2]) @ mat_rot_t
                        parts = bbox_pred_list[lvl].new_zeros((len(shift_xx), part_num), dtype=torch.float)
                        if self.cfg.part_type == 'up_down':
                            mask = (relative_shifts[:, 1] >=0).byte()
                            parts[mask, 0] = 1
                            mask = (relative_shifts[:, 1] < 0).byte()
                            parts[mask, 1] = 1
                        elif self.cfg.part_type == 'left_right':
                            mask = (relative_shifts[:, 0] >=0).byte()
                            parts[mask, 0] = 1
                            mask = (relative_shifts[:, 0] < 0).byte()
                            parts[mask, 1] = 1
                        elif self.cfg.part_type == 'quadrant':
                            mask = (relative_shifts[:, 0] >= 0).byte() & (relative_shifts[:, 1] >=0).byte()
                            parts[mask, 0] = 1
                            mask = (relative_shifts[:, 0] < 0).byte() & (relative_shifts[:, 1] >= 0).byte()
                            parts[mask, 1] = 1
                            mask = (relative_shifts[:, 0] < 0).byte() & (relative_shifts[:, 1] < 0).byte()
                            parts[mask, 2] = 1
                            mask = (relative_shifts[:, 0] >= 0).byte() & (relative_shifts[:, 1] < 0).byte()
                            parts[mask, 3] = 1
                        else:
                            mask = (relative_shifts[:, 0] >= 0).byte() & (relative_shifts[:, 1] >=0).byte()
                            mask1 = mask & (relative_shifts[:, 0] >= relative_shifts[:, 1]).byte()
                            mask2 = mask & (relative_shifts[:, 0] < relative_shifts[:, 1]).byte()
                            parts[mask1, 0] = 1
                            parts[mask2, 1] = 1
                            mask = (relative_shifts[:, 0] < 0).byte() & (relative_shifts[:, 1] >= 0).byte()
                            mask1 = mask & ((-relative_shifts[:, 0]) <= relative_shifts[:, 1]).byte()
                            mask2 = mask & ((-relative_shifts[:, 0]) > relative_shifts[:, 1]).byte()
                            parts[mask1, 2] = 1
                            parts[mask2, 3] = 1
                            mask = (relative_shifts[:, 0] < 0).byte() & (relative_shifts[:, 1] < 0).byte()
                            mask1 = mask & ((-relative_shifts[:, 0]) >= (-relative_shifts[:, 1])).byte()
                            mask2 = mask & ((-relative_shifts[:, 0]) < (-relative_shifts[:, 1])).byte()
                            parts[mask1, 4] = 1
                            parts[mask2, 5] = 1
                            mask = (relative_shifts[:, 0] >= 0).byte() & (relative_shifts[:, 1] < 0).byte()
                            mask1 = mask & (relative_shifts[:, 0] <= (-relative_shifts[:, 1])).byte()
                            mask2 = mask & (-relative_shifts[:, 0] > (-relative_shifts[:, 1])).byte()
                            parts[mask1, 6] = 1
                            parts[mask2, 7] = 1
                        _part_labels = torch.cat((_part_labels, parts), dim=0)
                    if self.cfg.centerness:
                        mat_rot_t = torch.tensor([[torch.cos(boxes[i, -1]), torch.sin(boxes[i, -1])],
                                             [-torch.sin(boxes[i, -1]), torch.cos(boxes[i, -1])]], device=device)
                        parts = (centers - boxes[i, 0:2]) @ mat_rot_t
                        parts[:, 0] = parts[:, 0]/boxes[i, 3]
                        parts[:, 1] = parts[:, 1]/boxes[i, 4]
                        _part_labels = torch.cat((_part_labels, parts), dim=0)
                    _bbox_targets = torch.cat((_bbox_targets, shifts), dim=0)
                    zeros = torch.zeros_like(locs_x)

                    locs = torch.stack((zeros, locs_y, locs_x), dim=-1)
                    _bbox_locs = torch.cat((_bbox_locs, locs), dim=0)
                    if self.use_iou_branch:
                        mask_iou = points_in_convex_polygon_torch(np.concatenate([ww+0.5, hh+0.5], 1), ignore_boxes_iou[i].unsqueeze(0))
                        # ig_ind_iou = mask_iou.nonzero()[:, 0]
                        # for k in ig_ind_iou:
                        #     if not k in pos_ind:
                        #         _gt_iou_weights[hh_l[k], ww_l[k]] = 0.0

                        mask = mask_iou & (~pos_mask)
                        ig_ind = mask.nonzero()[:, 0]
                        ig_h, ig_w = hh_l[ig_ind], ww_l[ig_ind]
                        _gt_iou_weights[ig_h, ig_w] = 0


            labels.append(_labels)
            label_weights.append(_label_weights)
            bbox_targets.append(_bbox_targets)
            bbox_locs.append(_bbox_locs)
            if self.cfg.part_classification or self.cfg.centerness:
                part_labels.append(_part_labels)
            if self.use_iou_branch:
                gt_iou_weights.append(_gt_iou_weights)

        num_pos = 0
        num_neg = 0
        for lvl in range(num_levels):
            npos = bbox_targets[lvl].size(0)
            num_pos += npos
            num_neg += label_weights[lvl].nonzero().size(0) - npos
        if not self.use_iou_branch:
            gt_iou_weights = [None]*len(labels)
        if (not self.cfg.part_classification) and (not self.cfg.centerness):
            part_labels = [None]*len(labels)
        return (
         labels, label_weights, bbox_targets, bbox_locs, num_pos,
         num_neg, gt_iou_weights, part_labels)


    def images_to_levels(self, target, num_imgs, num_levels, is_cls=True):
        level_target = []
        if is_cls:
            for lvl in range(num_levels):
                level_target.append(torch.stack([target[i][lvl] for i in range(num_imgs)], dim=0))

        else:
            for lvl in range(num_levels):
                level_target.append(torch.cat([target[j][lvl] for j in range(num_imgs)], dim=0))

        return level_target

    def xyxy2xcycwh(self, xyxy):
        """Convert [x1 y1 x2 y2] box format to [xc yc w h] format."""
        return torch.cat((
         0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]), dim=1)

    def xcycwh2xyxy(self, xywh):
        """Convert [xc yc w y] box format to [x1 y1 x2 y2] format."""
        return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4],
         xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), dim=1)

    def prop_box_bounds(self, prop_boxes, scale, width, height):
        """Compute proportional box regions.
        Box centers are fixed. Box w and h scaled by scale.
        """
        #prop_boxes[:, 2:] *= scale
        prop_boxes = self.xcycwh2xyxy(prop_boxes)
        x1 = prop_boxes[:, 0].int().clamp(0, width - 1)
        y1 = prop_boxes[:, 1].int().clamp(0, height - 1)
        x2 = torch.ceil(prop_boxes[:, 2]).clamp(1, width).int()
        y2 = torch.ceil(prop_boxes[:, 3]).clamp(1, height).int()
        return (x1, y1, x2, y2)
    def _meshgrid(self, x, y):
        xx = x.repeat(len(y))
        yy = y.contiguous().vipyew(-1, 1).repeat(1, len(x)).contiguous().view(-1)
        return xx, yy
