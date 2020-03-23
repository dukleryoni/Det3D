from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.models.detectors import ohs_helper
import torch
import time
from torch.nn import AdaptiveMaxPool2d
from det3d.models.losses.fsaf_loss import RefineMultiBoxFSAFLoss
from collections import defaultdict
import torch.nn.functional as F
from det3d.core import box_torch_ops
# from .ohs_helper import generate_points, limit_period
import numpy as np
import itertools



@DETECTORS.register_module
class VoxelNet_OHS(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        loss=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet_OHS, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained,
        )
        self.fsaf = 0
        self.eval_fsaf=0

        self.ohs=None
        if hasattr(neck,"ohs"):
            self.ohs = neck.ohs
        elif hasattr(bbox_head,"ohs"):
            self.ohs = bbox_head.ohs

        if self.ohs is not None:
            self.fsaf = self.ohs.fsaf
            self.eval_fsaf = self.ohs.fsaf
            self.encode_background_as_zeros = self.ohs.encode_background_as_zeros
            self.use_iou_branch = self.ohs.use_iou_branch
            self.fsaf_cfg = self.ohs.fsaf_module
            self.class_names = list(itertools.chain(*[t["class_names"] for t in self.fsaf_cfg.tasks]))
            self.num_class = len(self.class_names)
            self.fsaf_loss = RefineMultiBoxFSAFLoss(self.fsaf_cfg, self.num_class+1,
                                                   pc_range=self.fsaf_cfg.range,
                                                   encode_background_as_zeros=self.encode_background_as_zeros,
                                                   use_iou_branch=self.use_iou_branch)

            self._use_direction_classifier = self.bbox_head.use_direction_classifier
            self.use_sigmoid_score = self.bbox_head.use_sigmoid_score
            self.dir_offset = self.bbox_head.direction_offset
            
            self._use_rotate_nms = self.test_cfg.nms.use_rotate_nms
            self._multiclass_nms = self.test_cfg.nms.use_multi_class_nms
            self._nms_score_thresholds = [self.test_cfg.score_threshold]
            self._nms_pre_max_sizes = [self.test_cfg.nms.nms_pre_max_size]
            self._nms_post_max_sizes = [self.test_cfg.nms.nms_post_max_size]
            self._nms_iou_thresholds = [self.test_cfg.nms.nms_iou_threshold]
            self._num_direction_bins = 2

    def extract_feat(self, data,):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        spatial_features = x
        if self.with_neck:
            x = self.neck(x)

        if self.fsaf > 0:
            if self.fsaf_cfg.use_occupancy:
                import spconv
                occupancy = spconv.SparseConvTensor(
                    torch.ones((len(data["coors"]), 1), device=data["coors"].device, dtype=spatial_features.dtype), data["coors"].int(),
                    data["input_shape"][::-1], data["batch_size"]).dense().squeeze(1)
                occupancy = AdaptiveMaxPool2d(spatial_features.shape[-2:])(occupancy).detach()
                occupancy, _ = torch.max(occupancy, dim=1)
                x['occupancy'] = occupancy.byte()
        return x

    def extract_feat_voxelDrop(self, data,):
        input_features = self.reader(data["features"], data["num_voxels"])
        use_voxel_drop = False

        if hasattr(self.train_cfg, "gt_drop") or hasattr(self.train_cfg, "general_voxel_drop"):
            use_voxel_drop = self.train_cfg.gt_drop or self.train_cfg.general_voxel_drop

        if use_voxel_drop:
            # General input VoxelDrop
            voxel_general_dropout = self.train_cfg.general_voxel_drop
            p_dropout = self.train_cfg.drop_rate
            if voxel_general_dropout:
                # This the logical mask of entries we keep (hence 1 - p_dropout)
                dropout_mask = torch.bernoulli((1 - p_dropout) * torch.ones(len(input_features))).type(torch.bool)

            voxel_gt_dropout = self.train_cfg.gt_drop
            if voxel_gt_dropout:
                gt_dropout_masks = ohs_helper.get_gt_masks(data=data, range=self.fsaf_cfg.range)
                dropout_mask = ohs_helper.get_gt_dropout(drop_rate=p_dropout, features=input_features,
                                                                  coors=data["coors"],
                                                                  masks=gt_dropout_masks)

            input_features = input_features[dropout_mask]
            data["features"] = data["features"][dropout_mask]
            data["coors"] = data["coors"][dropout_mask]
            data["num_voxels"] = data["num_voxels"][dropout_mask]

        x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])

        spatial_features = x

        if self.with_neck:
            x = self.neck(x)

        if self.fsaf > 0:
            if self.fsaf_cfg.use_occupancy:
                import spconv
                occupancy = spconv.SparseConvTensor(
                    torch.ones((len(data["coors"]), 1), device=data["coors"].device, dtype=spatial_features.dtype),
                    data["coors"].int(),
                    data["input_shape"][::-1], data["batch_size"]).dense().squeeze(1)
                occupancy = AdaptiveMaxPool2d(spatial_features.shape[-2:])(occupancy).detach()
                occupancy, _ = torch.max(occupancy, dim=1)
                x['occupancy'] = occupancy.byte()

        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        if self.fsaf > 0 and return_loss:
            data["fsaf_targets"] = example["fsaf_targets"]

        preds_dict = self.extract_feat_voxelDrop(data)

        if self.fsaf  > 0:
            # mg_Head split
            if hasattr(self.bbox_head, "ohs"): # TODO go over this and see what would be the cleanest way to go
                preds = self.bbox_head(preds_dict)

                if return_loss:
                    return self.bbox_head.loss(example, preds)
                else:
                    with torch.no_grad():
                        return self.predict(example, preds) # need to replace with mg_split version, to add multiclass decide how to incorporate the loss

            else: #FSAF
                if return_loss:
                    return self.loss(example, preds_dict)

                else:
                    with torch.no_grad():
                        return self.predict(example, preds_dict)

        else: #regular
            preds = self.bbox_head(preds_dict)
            if return_loss:
                return self.bbox_head.loss(example, preds)
            else:
                return self.bbox_head.predict(example, preds, self.test_cfg)

    def loss(self, example, preds_dict):
        fsaf_targets = example["fsaf_targets"]
        loss = 0
        res = defaultdict(list)

        if self.fsaf > 0:
            if self.fsaf_cfg.use_occupancy:
                fsaf_loss_cls, fsaf_loss_bbox, fsaf_cls_score, fsaf_labels, fsaf_refinements = self.fsaf_loss(
                    preds_dict['fsaf'], fsaf_targets, occupancy=preds_dict['occupancy'])
            else:
                fsaf_loss_cls, fsaf_loss_bbox, fsaf_cls_score, fsaf_labels, fsaf_refinements = self.fsaf_loss(
                    preds_dict['fsaf'], fsaf_targets)
            if len(fsaf_loss_cls) == 1:
                fsaf_loss_cls = fsaf_loss_cls[0]
                fsaf_loss_bbox = fsaf_loss_bbox[0]
            else:
                fsaf_loss_cls = torch.cat(fsaf_loss_cls, 0)
                fsaf_loss_bbox = torch.cat(fsaf_loss_bbox, 0)
            batch_size_dev = len(fsaf_targets)

            loss += (fsaf_loss_cls + fsaf_loss_bbox)
            res["fsaf_loss_cls"].append(fsaf_loss_cls)
            res["fsaf_loss_box"].append(fsaf_loss_bbox)
            res["fsaf_labels"].append(fsaf_labels)
            res["fsaf_cls_preds"].append(fsaf_cls_score)
            res['loss'].append(loss)
        return res

    def predict_single(self, box_preds, cls_preds, dir_preds, features, a_mask, meta, feature_map_size_prod, dtype,
                       device, num_class_with_bg, post_center_range, predictions_dicts, points):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
        box_preds = box_preds.float()
        cls_preds = cls_preds.float()


        if self._use_direction_classifier and (dir_preds is not None):
            if a_mask is not None:
                dir_preds = dir_preds[a_mask]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
        if self.encode_background_as_zeros:
            # this don't support softmax
            assert self.use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
        else:
            # encode background as first element in one-hot vector
            if self.use_sigmoid_score:
                total_scores = torch.sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

        # Apply NMS in birdeye view
        if self._use_rotate_nms:
            nms_func = box_torch_ops.rotate_nms
        else:
            nms_func = box_torch_ops.nms
        if self._multiclass_nms:
            # Currently not functional if multiclass_nms need to declare a few more variables
            assert self.encode_background_as_zeros is True
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            if not self._use_rotate_nms:
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)

            selected_boxes, selected_labels, selected_scores = [], [], []
            selected_dir_labels = []

            boxes = boxes_for_nms
            selected_per_class = []
            score_threshs = self._nms_score_thresholds
            pre_max_sizes = self._nms_pre_max_sizes
            post_max_sizes = self._nms_post_max_sizes
            iou_thresholds = self._nms_iou_thresholds
            for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                    range(self.num_class),
                    score_threshs,
                    pre_max_sizes, post_max_sizes, iou_thresholds):
                if self._nms_class_agnostic:
                    class_scores = total_scores.view(
                        feature_map_size_prod, -1,
                        self.num_class)[..., class_idx]
                    class_scores = class_scores.contiguous().view(-1)
                    class_boxes_nms = boxes.view(-1,
                                                 boxes_for_nms.shape[-1])
                    class_boxes = box_preds
                    class_dir_labels = dir_labels
                else:
                    anchors_range = self.target_assigner.anchors_range(class_idx)
                    class_scores = total_scores.view(
                        -1,
                        self.num_class)[anchors_range[0]:anchors_range[1], class_idx]
                    class_boxes_nms = boxes.view(-1,
                                                 boxes_for_nms.shape[-1])[anchors_range[0]:anchors_range[1], :]
                    class_scores = class_scores.contiguous().view(-1)
                    class_boxes_nms = class_boxes_nms.contiguous().view(
                        -1, boxes_for_nms.shape[-1])
                    class_boxes = box_preds.view(-1,
                                                 box_preds.shape[-1])[anchors_range[0]:anchors_range[1], :]
                    class_boxes = class_boxes.contiguous().view(
                        -1, box_preds.shape[-1])
                    if self._use_direction_classifier and (dir_preds is not None):
                        class_dir_labels = dir_labels.view(-1)[anchors_range[0]:anchors_range[1]]
                        class_dir_labels = class_dir_labels.contiguous(
                        ).view(-1)
                if score_thresh > 0.0:
                    class_scores_keep = class_scores >= score_thresh
                    if class_scores_keep.shape[0] == 0:
                        selected_per_class.append(None)
                        continue
                    class_scores = class_scores[class_scores_keep]
                if class_scores.shape[0] != 0:
                    if score_thresh > 0.0:
                        class_boxes_nms = class_boxes_nms[
                            class_scores_keep]
                        class_boxes = class_boxes[class_scores_keep]
                        class_dir_labels = class_dir_labels[
                            class_scores_keep]
                    keep = nms_func(class_boxes_nms, class_scores, pre_ms,
                                    post_ms, iou_th)
                    if keep.shape[0] != 0:
                        selected_per_class.append(keep)
                    else:
                        selected_per_class.append(None)
                else:
                    selected_per_class.append(None)
                selected = selected_per_class[-1]

                if selected is not None:
                    selected_boxes.append(class_boxes[selected])
                    selected_labels.append(
                        torch.full([class_boxes[selected].shape[0]],
                                   class_idx,
                                   dtype=torch.int64,
                                   device=box_preds.device))
                    if self._use_direction_classifier and (dir_preds is not None):
                        selected_dir_labels.append(
                            class_dir_labels[selected])
                    selected_scores.append(class_scores[selected])
            selected_boxes = torch.cat(selected_boxes, dim=0)
            selected_labels = torch.cat(selected_labels, dim=0)
            selected_scores = torch.cat(selected_scores, dim=0)
            if self._use_direction_classifier and (dir_preds is not None):
                selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
        else:
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(
                    total_scores, dim=-1)
            if self._nms_score_thresholds[0] > 0.0:
                top_scores_keep = top_scores >= self._nms_score_thresholds[0]
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self._nms_score_thresholds[0] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self._use_direction_classifier and (dir_preds is not None):
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]


                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_sizes[0],
                    post_max_size=self._nms_post_max_sizes[0],
                    iou_threshold=self._nms_iou_thresholds[0],
                )
            else:
                selected = []
            # if selected is not None:
            selected_boxes = box_preds[selected]
            if self._use_direction_classifier and (dir_preds is not None):
                selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

        # finally generate predictions.
        if selected_boxes.shape[0] != 0:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            yaw = box_preds[..., 6]
            if self._use_direction_classifier and (dir_preds is not None):
                dir_labels = selected_dir_labels
                period = (2 * np.pi / self._num_direction_bins)
                # dir_rot = box_torch_ops.limit_period(S
                #     box_preds[..., 6] - self.dir_offset,
                #     self._dir_limit_offset, period)
                dir_rot = ohs_helper.limit_period(
                    box_preds[..., -1] - self.dir_offset,  # changed for velocity
                    0, 2 * np.pi)
                yaw = dir_rot + self.dir_offset + period * dir_labels


          #  box_preds_vel_rot = torch.cat((box_preds[..., -3:-1], yaw), -1).type(dtype=box_preds.dtype)  #TODO Add velocity to predictions, for now it is 0

          #   box_preds_vel_rot = torch.stack(torch.distributions.utils.broadcast_all(0, 0, yaw),-1).type(dtype=box_preds.dtype)  # TODO Add velocity to predictions, for now it is 0
            box_preds = torch.cat((box_preds[..., :-1], yaw.view(-1,1)), -1) # Changed from 6 to -1 to add velocity

            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            if post_center_range is not None:
                mask = (final_box_preds[:, :3] >=
                        post_center_range[:3]).all(1)
                mask &= (final_box_preds[:, :3] <=
                         post_center_range[3:]).all(1)
                predictions_dict = {
                    "box3d_lidar": final_box_preds[mask],
                    "scores": final_scores[mask],
                    "label_preds": label_preds[mask],
                    "metadata": meta,
                }
            else:
                predictions_dict = {
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "metadata": meta,
                }
        else:
            predictions_dict = {
                "box3d_lidar":
                    torch.zeros([0, box_preds.shape[-1]+2],    # Added to take care of velocity
                                dtype=dtype,
                                device=device),
                "scores":
                    torch.zeros([0], dtype=dtype, device=device),
                "label_preds":
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                "metadata":
                    meta,
            }
        '''
        for i in range(8000):
            if not os.path.exists('npys/total_scores_%i.npy'%i):
                np.save('npys/total_scores_%i.npy'%i, total_scores.cpu().numpy())
                try:
                    np.save('npys/final_box_preds_%i.npy'%i, final_box_preds.cpu().numpy())
                    np.save('npys/final_scores_%i.npy'%i, final_scores.cpu().numpy())
                    np.save('npys/label_preds_%i.npy'%i, label_preds.cpu().numpy())
                except:
                    shabi=1
                break
        '''
        predictions_dicts.append(predictions_dict)

    def predict(self, example, preds_dict):
        """
        start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_dict.
            pred_dict: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                metadata: meta-data which contains dataset-specific information.
                    for kitti, it contains image idx (label idx),
                    for nuscenes, sample_token is saved in it.
            }
        """
        meta_list = example["metadata"]
        batch_size = len(meta_list)
        t = time.time()
        num_class_with_bg = self.num_class
        if not self.encode_background_as_zeros:
            num_class_with_bg = self.num_class + 1
        predictions_dicts = []
        post_center_range = None
        if "post_center_range" in self.__dir__() and len(self._post_center_range) > 0:
            post_center_range = torch.tensor(
                self._post_center_range,
                dtype=torch.float32,
                device=torch.cuda.current_device()).float()
        if (self.fsaf > 0) and (self.eval_fsaf > 0):  # currently only support one level
            fsaf_batch_cls_preds, fsaf_batch_box_preds = preds_dict['fsaf'][0][0], preds_dict['fsaf'][1][0]
            points = ohs_helper.generate_points(fsaf_batch_cls_preds.shape[-2:], np.asarray(self.fsaf_cfg.range),
                                     fsaf_batch_cls_preds.device)

            fsaf_batch_box_preds = fsaf_batch_box_preds.permute(0, 2, 3, 1)
            fsaf_batch_cls_preds = fsaf_batch_cls_preds.permute(0, 2, 3, 1)
            fsaf_batch_box_preds = fsaf_batch_box_preds.view(batch_size, -1, fsaf_batch_box_preds.shape[-1])
            fsaf_batch_cls_preds = fsaf_batch_cls_preds.view(batch_size, -1, num_class_with_bg)
            if self.use_iou_branch:
                fsaf_batch_iou_preds = preds_dict['fsaf'][2][0]

            '''
            if self.FSAFLoss.cfg.centerness:
                fsaf_batch_centerness_preds = preds_dict['fsaf'][3][0]
                fsaf_batch_centerness_preds = fsaf_batch_centerness_preds.permute(0, 2, 3, 1)
                fsaf_batch_centerness_preds = fsaf_batch_centerness_preds.view(batch_size, -1, fsaf_batch_centerness_preds.shape[-1])
                fsaf_batch_centerness_preds = torch.exp(-0.2*fsaf_batch_centerness_preds.norm(dim=-1, keepdim=True))
            if fsaf_batch_iou_preds is not None:
                fsaf_batch_iou_preds = fsaf_batch_iou_preds.permute(0, 2, 3, 1)
                fsaf_batch_iou_preds = fsaf_batch_iou_preds.view(batch_size, -1, fsaf_batch_iou_preds.shape[-1])
                fsaf_batch_cls_preds *= fsaf_batch_iou_preds
            '''
            if 'center' in self.fsaf_cfg.loc_type:
                # centers_2d = ((batch_box_preds[:, :, :2] - 0.5)*batch_box_preds[:, :, 3:5]).contiguous().view(-1, 1, 2)
                if self.fsaf_cfg.rot_type == 'cos_sin':
                    '''
                    rot_mat_T = torch.stack(
                         [tstack([batch_box_preds[:, :, 6].contiguous().view(-1), -batch_box_preds[:, :, 7].contiguous().view(-1)]),
                          tstack([batch_box_preds[:, :, 7].contiguous().view(-1), batch_box_preds[:, :, 6].contiguous().view(-1)])])
                    centers_2d = torch.einsum('aij,jka->aik', (centers_2d, rot_mat_T)).view(batch_size, -1, 2)
                    '''
                    fsaf_batch_box_preds[:, :, -2] = torch.atan2(fsaf_batch_box_preds[:, :, -1],
                                                                fsaf_batch_box_preds[:, :, -2])
                    fsaf_batch_box_preds = fsaf_batch_box_preds[:, :, :-1]
                if self.fsaf_cfg.rot_type == 'softbin_cos_sin':
                    fsaf_batch_box_preds = fsaf_batch_box_preds[:, :, :-1]
                # else:
                # batch_box_preds[:, :, :2] = points[:, :2] - centers_2d
                fsaf_batch_box_preds[:, :, :2] += points[:, :2]

                fsaf_batch_box_preds[:, :, 3:6] = torch.exp(fsaf_batch_box_preds[:, :, 3:6])
                if 'bottom' in self.fsaf_cfg.h_loc_type:
                    fsaf_batch_box_preds[:, :, 2] += 0.5 * fsaf_batch_box_preds[:, :, 5]
            else:
                print("We should not be here")
                fsaf_batch_box_preds *= self.fsaf_cfg.norm_factor
                centers_2d = 0.5 * torch.cat([-fsaf_batch_box_preds[:, :, 0:1] + fsaf_batch_box_preds[:, :, 3:4],
                                              fsaf_batch_box_preds[:, :, 1:2] - fsaf_batch_box_preds[:, :, 4:5]],
                                             -1).reshape(-1, 1, 2)
                centers_2d = box_torch_ops.rotation_2d(centers_2d, fsaf_batch_box_preds[:, :, -1].reshape(-1)).reshape(
                    batch_size, -1, 2)
                centers_2d += points[:, :2]
                dims = fsaf_batch_box_preds[:, :, :3] + fsaf_batch_box_preds[:, :, 3:6]
                fsaf_batch_box_preds[:, :, :2] = centers_2d
                fsaf_batch_box_preds[:, :, 2] = 0.5 * (
                            points[:, 2] - fsaf_batch_box_preds[:, :, 2] + points[:, 2] + fsaf_batch_box_preds[:, :, 5])
                fsaf_batch_box_preds[:, :, 3:6] = dims
            # batch_box_preds[:, :, 6] = 0
        if self.eval_fsaf == 1:
            batch_anchors_mask = [None] * batch_size
            num_anchors_per_location = 1
            # print('box_preds', batch_box_preds)
            batch_box_preds = fsaf_batch_box_preds
            batch_cls_preds = fsaf_batch_cls_preds
            '''
            if self.FSAFLoss.cfg.centerness:
                batch_centerness_preds = fsaf_batch_centerness_preds
            '''
            batch_dir_preds = [None] * batch_size
            feature_map_size_prod = batch_box_preds.shape[1]

        batch_features = [None] * batch_size
        points = None
        for box_preds, cls_preds, dir_preds, features, a_mask, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds,
                batch_features, batch_anchors_mask, meta_list):
            self.predict_single(box_preds, cls_preds, dir_preds, features, a_mask, meta, feature_map_size_prod,
                                batch_box_preds.dtype, batch_box_preds.device, num_class_with_bg, post_center_range,
                                predictions_dicts, points)
        return predictions_dicts