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
#from .ohs_helper import generate_points, limit_period
import numpy as np



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
        if neck.get("ohs") is not None:
            self.fsaf = neck.ohs.fsaf
            self.eval_fsaf = neck.ohs.fsaf
            self.encode_background_as_zeros = neck.ohs.encode_background_as_zeros
            self.use_iou_branch = neck.ohs.use_iou_branch
            self.fsaf_cfg = neck.ohs.fsaf_module
            self.num_class = len(self.fsaf_cfg.tasks[0]["class_names"])
            self.fsaf_loss = RefineMultiBoxFSAFLoss(self.fsaf_cfg, self.num_class+1,
                                                   pc_range=self.fsaf_cfg.range,
                                                   encode_background_as_zeros=self.encode_background_as_zeros,
                                                   use_iou_branch=self.use_iou_branch)

            self._use_direction_classifier = self.bbox_head.use_direction_classifier
            self._use_sigmoid_score = self.bbox_head.use_sigmoid_score
            self._use_rotate_nms = self.test_cfg.nms.use_rotate_nms
            self._multiclass_nms = self.test_cfg.nms.use_multi_class_nms
            self._nms_score_thresholds = [self.test_cfg.score_threshold]
            self._nms_pre_max_sizes = [self.test_cfg.nms.nms_pre_max_size]
            self._nms_post_max_sizes = [self.test_cfg.nms.nms_post_max_size]
            self._nms_iou_thresholds = [self.test_cfg.nms.nms_iou_threshold]
            self._num_direction_bins = 2
            self._dir_offset = self.bbox_head.direction_offset

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

    def loss(self, example, preds_dict, fsaf_targets=None):
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

        preds_dict = self.extract_feat(data)
        if self.fsaf  > 0:
            if return_loss:
                fsaf_new_loss = self.loss(example, preds_dict, example['fsaf_targets'])
                return fsaf_new_loss
            else:
                with torch.no_grad():
                    blob = self.predict(example, preds_dict)
                    return blob

        # x = preds_dict["out"]
        preds = self.bbox_head(preds_dict)
        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

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
            assert self._use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
        else:
            # encode background as first element in one-hot vector
            if self._use_sigmoid_score:
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
                #     box_preds[..., 6] - self._dir_offset,
                #     self._dir_limit_offset, period)
                dir_rot = ohs_helper.limit_period(
                    box_preds[..., 6] - self._dir_offset,
                    0, 2 * np.pi)
                yaw = dir_rot + self._dir_offset + period * dir_labels


            box_preds_vel_rot = torch.stack(torch.distributions.utils.broadcast_all(0, 0, yaw),-1).type(dtype=box_preds.dtype)  # Add velocity to predictions, for now it is 0
            box_preds = torch.cat((box_preds[..., :6], box_preds_vel_rot), -1)

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
                    fsaf_batch_box_preds[:, :, 6] = torch.atan2(fsaf_batch_box_preds[:, :, 7],
                                                                fsaf_batch_box_preds[:, :, 6])
                    fsaf_batch_box_preds = fsaf_batch_box_preds[:, :, :7]
                if self.fsaf_cfg.rot_type == 'softbin_cos_sin':
                    fsaf_batch_box_preds = fsaf_batch_box_preds[:, :, :7]
                # else:
                # batch_box_preds[:, :, :2] = points[:, :2] - centers_2d
                fsaf_batch_box_preds[:, :, :2] += points[:, :2]

                fsaf_batch_box_preds[:, :, 3:6] = torch.exp(fsaf_batch_box_preds[:, :, 3:6])
                if 'bottom' in self.fsaf_cfg.h_loc_type:
                    fsaf_batch_box_preds[:, :, 2] += 0.5 * fsaf_batch_box_preds[:, :, 5]
            else:
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

    # def network_forward(self, voxels, roi_features, num_points, coors, batch_size, fusion_type, num_voxels, grid_size):
    #
    #     voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
    #     if fusion_type is not None and 'voxel' in fusion_type:
    #         print('fusion')
    #         voxel_features = torch.cat([voxel_features, roi_features], dim=-1)
    #
    #     spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size)
    #     preds_dict = self.rpn(spatial_features)
    #
    #     if self.fsaf > 0 and self.FSAFLoss.cfg.use_occupancy:
    #         import spconv
    #         occupancy = spconv.SparseConvTensor(
    #             torch.ones((len(coors), 1), device=coors.device, dtype=spatial_features.dtype), coors.int(),
    #             grid_size[::-1], batch_size).dense().squeeze(1)
    #         occupancy = nn.AdaptiveMaxPool2d(spatial_features.shape[-2:])(occupancy).detach()
    #         occupancy, _ = torch.max(occupancy, dim=1)
    #         preds_dict['occupancy'] = occupancy.byte()
    #     return preds_dict
    #
    # def forward(self, example):
    #     """module's forward should always accept dict and return loss.
    #     """
    #     device = torch.cuda.current_device()
    #     voxels = example["voxels"]
    #     num_points = example["num_points"]
    #     coors = example["coordinates"]
    #
    #     if len(num_points.shape) == 2:  # multi-gpu
    #         print("here in wierd state")
    #         num_voxel_per_batch = example["num_voxels"].cpu().numpy().reshape(-1)
    #         voxel_list = []
    #         num_points_list = []
    #         coors_list = []
    #         for i, num_voxel in enumerate(num_voxel_per_batch):
    #             voxel_list.append(voxels[i, :num_voxel])
    #             num_points_list.append(num_points[i, :num_voxel])
    #             coors_list.append(coors[i, :num_voxel])
    #         voxels = torch.cat(voxel_list, dim=0)
    #         num_points = torch.cat(num_points_list, dim=0)
    #         coors = torch.cat(coors_list, dim=0)
    #     batch_anchors = example["anchors"]
    #     batch_size_dev = batch_anchors.shape[0]
    #     # features: [num_voxels, max_num_points_per_voxel, 7]
    #     # num_points: [num_voxels]
    #     # coors: [num_voxels, 4]
    #
    #     roi_features_batch = None  # remove
    #     # remove
    #     preds_dict = self.network_forward(voxels, roi_features_batch, num_points, coors, batch_size_dev,
    #                                       self.fusion_type, example['num_voxels'], example['grid_size'][0])
    #
    #     if self.training:
    #         if self.fsaf > 0:
    #             res, fsaf_refinements = self.loss(example, preds_dict, example['fsaf_targets'])
    #             # if self.FSAFLoss.cfg.refinement:
    #             #     print('refinement')
    #             #     assert batch_size_dev == 1, "currently only support refinement for batch_size_dev == 1"
    #             #     if len(fsaf_refinements['bbox_pred'])==0:
    #             #         refine_loss = self.refine(noise_rpn_features=fsaf_refinements['noise_feature_list'][0])
    #             #     else:
    #             #         refine_loss = self.refine(example['points'][0], fsaf_refinements['bbox_pred'][0], fsaf_refinements['feature_list'][0], fsaf_refinements['bbox_target'][0], fsaf_refinements['noise_feature_list'][0])
    #             #     res['fsaf_loss_box'] += refine_loss
    #             #     res['loss'] += refine_loss
    #             #     print(refine_loss)
    #             return res
    #         return self.loss(example, preds_dict)
    #
    #     else:  # evaluate
    #         self.start_timer("predict")
    #         with torch.no_grad():
    #             res = self.predict(example, preds_dict)
    #         self.end_timer("predict")
    #         return res

# class VoxelNet(nn.Module):
#     def __init__(self,
#                  output_shape,
#                  num_class=2,
#                  num_input_features=4,
#                  vfe_class_name="VoxelFeatureExtractor",
#                  vfe_num_filters=[32, 128],
#                  with_distance=False,
#                  middle_class_name="SparseMiddleExtractor",
#                  middle_num_input_features=-1,
#                  middle_num_filters_d1=[64],
#                  middle_num_filters_d2=[64, 64],
#                  rpn_class_name="RPN",
#                  rpn_num_input_features=-1,
#                  rpn_layer_nums=[3, 5, 5],
#                  rpn_layer_strides=[2, 2, 2],
#                  rpn_num_filters=[128, 128, 256],
#                  rpn_upsample_strides=[1, 2, 4],
#                  rpn_num_upsample_filters=[256, 256, 256],
#                  use_norm=True,
#                  use_groupnorm=False,
#                  num_groups=32,
#                  use_direction_classifier=True,
#                  use_sigmoid_score=False,
#                  encode_background_as_zeros=True,
#                  use_rotate_nms=True,
#                  multiclass_nms=False,
#                  nms_score_thresholds=None,
#                  nms_pre_max_sizes=None,
#                  nms_post_max_sizes=None,
#                  nms_iou_thresholds=None,
#                  target_assigner=None,
#                  cls_loss_weight=1.0,
#                  loc_loss_weight=1.0,
#                  pos_cls_weight=1.0,
#                  neg_cls_weight=1.0,
#                  direction_loss_weight=1.0,
#                  loss_norm_type=LossNormType.NormByNumPositives,
#                  encode_rad_error_by_sin=False,
#                  loc_loss_ftor=None,
#                  cls_loss_ftor=None,
#                  measure_time=False,
#                  voxel_generator=None,
#                  post_center_range=None,
#                  dir_offset=0.0,
#                  sin_error_factor=1.0,
#                  nms_class_agnostic=False,
#                  num_direction_bins=2,
#                  direction_limit_offset=0,
#                  name='voxelnet',
#                  fusion_type=None,
#                  frcnn_cfg=None,
#                  fsaf=0,
#                  fsaf_cfg=None,
#                  use_iou_branch=False,
#                  iou_loss_weight=1,
#                  eval_fsaf=0):
#         super().__init__()
#         self.name = name
#         self._sin_error_factor = sin_error_factor
#         self.num_class = num_class
#         self._use_rotate_nms = use_rotate_nms
#         self._multiclass_nms = multiclass_nms
#         self._nms_score_thresholds = nms_score_thresholds
#         self._nms_pre_max_sizes = nms_pre_max_sizes
#         self._nms_post_max_sizes = nms_post_max_sizes
#         self._nms_iou_thresholds = nms_iou_thresholds
#         self._use_sigmoid_score = use_sigmoid_score
#         self.encode_background_as_zeros = encode_background_as_zeros
#         self._use_direction_classifier = use_direction_classifier
#         self._num_input_features = num_input_features
#         self._box_coder = target_assigner.box_coder
#         self.target_assigner = target_assigner
#         self.voxel_generator = voxel_generator
#         self._pos_cls_weight = pos_cls_weight
#         self._neg_cls_weight = neg_cls_weight
#         self._encode_rad_error_by_sin = encode_rad_error_by_sin
#         self._loss_norm_type = loss_norm_type
#         self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
#         self._diff_loc_loss_ftor = WeightedSmoothL1LocalizationLoss()
#         self._dir_offset = dir_offset
#         self._loc_loss_ftor = loc_loss_ftor
#         self._cls_loss_ftor = cls_loss_ftor
#         self._direction_loss_weight = direction_loss_weight
#         self._cls_loss_weight = cls_loss_weight
#         self._loc_loss_weight = loc_loss_weight
#         self._post_center_range = post_center_range or []
#         self.measure_time = measure_time
#         self._nms_class_agnostic = nms_class_agnostic
#         self._num_direction_bins = num_direction_bins
#         self._dir_limit_offset = direction_limit_offset
#         self.dataset = None
#         self.fsaf = fsaf
#         self.use_iou_branch = use_iou_branch
#         self.iou_loss = WeightedSmoothL1LocalizationLoss(1.0)
#         self.iou_loss_weight = iou_loss_weight
#         self.eval_fsaf = eval_fsaf
#
#         if self.fsaf > 0:
#             from second.lib.losses.refine_multibox_fsaf_loss import RefineMultiBoxFSAFLoss
#             if isinstance(voxel_generator, list):
#                 grid_size = voxel_generator[0].grid_size
#                 pc_range = voxel_generator[0].point_cloud_range
#             else:
#                 grid_size = voxel_generator.grid_size
#                 pc_range = voxel_generator.point_cloud_range
#             self.FSAFLoss = RefineMultiBoxFSAFLoss(fsaf_cfg, self.num_class + 1,
#                                                    (grid_size[1], grid_size[0]),
#                                                    pc_range,
#                                                    encode_background_as_zeros,
#                                                    use_iou_branch=use_iou_branch)
#             if fsaf_cfg.refinement:
#                 self.fsaf_refiner = Refiner(fsaf_cfg.sparse_shape[::-1], middle_num_input_features,
#                                             sum(rpn_num_upsample_filters), fsaf_cfg.refinement_group_norm_num)
#                 self.refine_voxel_feature_extractor = voxel_encoder.get_vfe_class(vfe_class_name)(
#                     num_input_features,
#                     use_norm,
#                     num_filters=vfe_num_filters,
#                     with_distance=with_distance,
#                     voxel_size=self.voxel_generator.voxel_size,
#                     pc_range=self.voxel_generator.point_cloud_range,
#                     name='refine_vfe',
#                 )
#         # RGB Branch
#         self.fusion_type = fusion_type
#         # if self.fusion_type == 'point_feature':
#         #     self.new_reduce_layer1 = nn.Conv1d(720, 32, 1)
#         #     self.new_reduce_layer2 = nn.Conv1d(32, 4, 1)
#         #     num_input_features+= 4
#         #     middle_num_input_features += 4
#         # if self.fusion_type == 'voxel':
#         #     from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
#         #     self.feature_extractor = make_roi_box_feature_extractor(frcnn_cfg, 720)
#         #     middle_num_input_features += 4
#         #     self.new_reduce_layer1 = nn.Conv2d(720, 32, 1)
#         #     self.new_reduce_layer2 = nn.Conv2d(32, 4, 1)
#
#         im_num_filters = None
#
#         if isinstance(voxel_generator, list):
#             self.voxel_feature_extractor = []
#             self.middle_feature_extractor = []
#             for i in range(2):
#                 self.voxel_feature_extractor.append(voxel_encoder.get_vfe_class(vfe_class_name)(
#                     num_input_features,
#                     use_norm,
#                     num_filters=vfe_num_filters,
#                     with_distance=with_distance,
#                     voxel_size=self.voxel_generator[i].voxel_size,
#                     pc_range=self.voxel_generator[i].point_cloud_range,
#                 )
#                 )
#         else:
#             self.voxel_feature_extractor = voxel_encoder.get_vfe_class(vfe_class_name)(
#                 num_input_features,
#                 use_norm,
#                 num_filters=vfe_num_filters,
#                 with_distance=with_distance,
#                 voxel_size=self.voxel_generator.voxel_size,
#                 pc_range=self.voxel_generator.point_cloud_range,
#             )
#         self.middle_feature_extractor = middle.get_middle_class(middle_class_name)(
#             output_shape,
#             use_norm,
#             num_input_features=middle_num_input_features,
#             num_filters_down1=middle_num_filters_d1,
#             num_filters_down2=middle_num_filters_d2)
#
#         self.rpn = rpn.get_rpn_class(rpn_class_name)(
#             use_norm=True,
#             num_class=num_class,
#             layer_nums=rpn_layer_nums,
#             layer_strides=rpn_layer_strides,
#             num_filters=rpn_num_filters,
#             upsample_strides=rpn_upsample_strides,
#             num_upsample_filters=rpn_num_upsample_filters,
#             num_input_features=rpn_num_input_features,
#             num_anchor_per_loc=target_assigner.num_anchors_per_location,
#             encode_background_as_zeros=encode_background_as_zeros,
#             use_direction_classifier=use_direction_classifier,
#             use_groupnorm=use_groupnorm,
#             num_groups=num_groups,
#             box_code_size=target_assigner.box_coder.code_size,
#             num_direction_bins=self._num_direction_bins,
#             im_num_filters=im_num_filters,
#             fsaf=fsaf,
#             fsaf_cfg=fsaf_cfg,
#             use_iou_branch=use_iou_branch)
#         self.rpn_acc = metrics.Accuracy(
#             dim=-1, encode_background_as_zeros=encode_background_as_zeros)
#         self.rpn_precision = metrics.Precision(dim=-1)
#         self.rpn_recall = metrics.Recall(dim=-1)
#         self.rpn_metrics = metrics.PrecisionRecall(
#             dim=-1,
#             thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
#             use_sigmoid_score=use_sigmoid_score,
#             encode_background_as_zeros=encode_background_as_zeros)
#
#         self.rpn_cls_loss = metrics.Scalar()
#         self.rpn_loc_loss = metrics.Scalar()
#         self.rpn_total_loss = metrics.Scalar()
#         self.register_buffer("global_step", torch.LongTensor(1).zero_())
#
#         self._time_dict = {}
#         self._time_total_dict = {}
#         self._time_count_dict = {}
#
#     def set_dataset(self, dataset):
#         self.dataset = dataset
#
#     def set_box_lidars(self, ngpus):
#         self.box_lidar = [None] * ngpus
#
#     def set_box_lidar(self, pc_range, voxel_size):
#         x = int(voxel_size[0] // 8)
#         y = int(voxel_size[1] // 8)
#         x0, y0, z0, x1, y1, z1 = pc_range
#         boxes_lidar = torch.zeros((x, y, 7), dtype=torch.float, device=torch.cuda.current_device())
#         h = (x1 - x0) / x
#         w = (y1 - y0) / y
#         boxes_lidar[:, :, 3] = h
#         boxes_lidar[:, :, 4] = w
#         boxes_lidar[:, :, 5] = float(z1 - z0)
#         boxes_lidar[:, :, 6] = 0
#         for i in range(x):
#             for j in range(y):
#                 boxes_lidar[i, j, 0] = x0 + h / 2 + h * i
#                 boxes_lidar[i, j, 1] = y0 + w / 2 + w * j
#                 boxes_lidar[i, j, 2] = float(z0)
#         self.box_lidar[torch.cuda.current_device()] = boxes_lidar.view(-1, 7)
#         self.box_h = x
#         self.box_w = y
#
#     def start_timer(self, *names):
#         if not self.measure_time:
#             return
#         torch.cuda.synchronize()
#         for name in names:
#             self._time_dict[name] = time.time()
#
#     def end_timer(self, name):
#         if not self.measure_time:
#             return
#         torch.cuda.synchronize()
#         time_elapsed = time.time() - self._time_dict[name]
#         if name not in self._time_count_dict:
#             self._time_count_dict[name] = 1
#             self._time_total_dict[name] = time_elapsed
#         else:
#             self._time_count_dict[name] += 1
#             self._time_total_dict[name] += time_elapsed
#         self._time_dict[name] = 0
#
#     def clear_timer(self):
#         self._time_count_dict.clear()
#         self._time_dict.clear()
#         self._time_total_dict.clear()
#
#     @contextlib.contextmanager
#     def profiler(self):
#         old_measure_time = self.measure_time
#         self.measure_time = True
#         yield
#         self.measure_time = old_measure_time
#
#     def get_avg_time_dict(self):
#         ret = {}
#         for name, val in self._time_total_dict.items():
#             count = self._time_count_dict[name]
#             ret[name] = val / max(1, count)
#         return ret
#
#     def update_global_step(self):
#         self.global_step += 1
#
#     def get_global_step(self):
#         return int(self.global_step.cpu().numpy()[0])
#
#     def clear_global_step(self):
#         self.global_step.zero_()
#
#     # def refine(self, points=None, bbox_preds=None, rpn_features=None, bbox_targets=None, noise_rpn_features=None):
#     #     print('we refine')
#     #     count = 0
#     #     if bbox_preds is not None:
#     #         if bbox_preds.shape[-1] == 8:
#     #             boxes_2d = center_to_corner_box2d(bbox_preds[:, :2], bbox_preds[:, 3:5], bbox_preds[:, 6], bbox_preds[:, 7])
#     #         else:
#     #             boxes_2d = center_to_corner_box2d(bbox_preds[:, :2], bbox_preds[:, 3:5], torch.cos(bbox_preds[:, 6]), torch.sin(bbox_preds[:, 6]))
#     #         xs_min, _ = boxes_2d[:, :, 0].min(1)
#     #         xs_max, _ = boxes_2d[:, :, 0].max(1)
#     #         ys_min, _ = boxes_2d[:, :, 1].min(1)
#     #         ys_max, _ = boxes_2d[:, :, 1].max(1)
#     #         voxel_features_batch = torch.zeros((0, self.FSAFLoss.cfg.voxel_generator.max_number_of_points_per_voxel, 4))
#     #         coors_batch = torch.zeros((0, 4), dtype=torch.int)
#     #         num_points_batch = torch.zeros(0, dtype=torch.int)
#     #         rpn_features_batch = rpn_features.new_zeros((0, rpn_features.shape[-1]))
#     #         mask = torch.zeros(len(bbox_preds), device=bbox_preds.device, dtype=torch.uint8)
#     #         if bbox_targets is None:
#     #             bbox_targets = [None] * len(bbox_preds)
#     #             bbox_targets_batch = None
#     #         else:
#     #             # bbox_targets_batch = []
#     #             bbox_targets_batch = bbox_targets.new_zeros((0, bbox_targets.shape[-1]))
#     #         for i, x_min, x_max, y_min, y_max, z_min, z_max, rpn_feat, bbox_target in zip(range(len(bbox_preds)), xs_min, xs_max, ys_min, ys_max, bbox_preds[:, 2]-0.5*bbox_preds[:, 5], bbox_preds[:, 2]+0.5*bbox_preds[:, 5], rpn_features, bbox_targets):
#     #             dx = x_max - x_min
#     #             dy = y_max - y_min
#     #             dz = z_max - z_min
#     #             x_min = x_min - self.FSAFLoss.cfg.refine_dilation * dx
#     #             x_max = x_max + self.FSAFLoss.cfg.refine_dilation * dx
#     #             y_min = y_min - self.FSAFLoss.cfg.refine_dilation * dy
#     #             y_max = y_max + self.FSAFLoss.cfg.refine_dilation * dy
#     #             z_min = z_min - self.FSAFLoss.cfg.refine_dilation * dz
#     #             z_max = z_max + self.FSAFLoss.cfg.refine_dilation * dz
#     #             self.FSAFLoss.cfg.voxel_generator.point_cloud_range[:] = [x_min, y_min, z_min, x_max, y_max, z_max]
#     #             self.FSAFLoss.cfg.voxel_generator.voxel_size[:] = [dx/self.FSAFLoss.cfg.sparse_shape[0], dy/self.FSAFLoss.cfg.sparse_shape[1], dz*1.2/self.FSAFLoss.cfg.sparse_shape[2]]
#     #             refinement_voxel_generator = voxel_builder.build(self.FSAFLoss.cfg.voxel_generator)
#     #             rets = refinement_voxel_generator.generate(points, 8000)
#     #             if rets['voxel_num'] <= 3: continue
#     #             voxel_features_batch = torch.cat([voxel_features_batch, torch.tensor(rets['voxels'])])
#     #             coors = torch.tensor(rets['coordinates'])
#     #             coors = torch.cat([coors.new_ones(len(coors), 1) * count, coors], 1)
#     #             coors_batch = torch.cat([coors_batch, coors.int()])
#     #             num_points_batch = torch.cat([num_points_batch, torch.tensor(rets['num_points_per_voxel']).int()])
#     #             rpn_features_batch = torch.cat([rpn_features_batch, rpn_feat.unsqueeze(0)])
#     #             if self.training:
#     #                 bbox_targets_batch = torch.cat([bbox_targets_batch, bbox_target.unsqueeze(0)])
#     #             mask[i] = 1
#     #             count += 1
#     #         if count > 0:
#     #             voxel_features_batch = voxel_features_batch.to(bbox_preds.device)
#     #             num_points_batch = num_points_batch.to(num_points_batch.device)
#     #             if self.global_step >= self.FSAFLoss.cfg.refinement_step:
#     #                 voxel_features_batch = self.FSAFLoss.cfg.refine_weight * self.refine_voxel_feature_extractor(voxel_features_batch, num_points_batch, coors_batch)
#     #             else:
#     #                 voxel_features_batch = 0 * self.refine_voxel_feature_extractor(voxel_features_batch, num_points_batch, coors_batch)
#     #     if count > 0:
#     #         refined_preds = self.fsaf_refiner(voxel_features_batch, coors_batch, count, rpn_features_batch)
#     #         if self.training:
#     #             refine_loss = self._loc_loss_ftor(refined_preds, bbox_targets_batch, weights=refined_preds.new_ones(len(refined_preds))).sum()/len(refined_preds)
#     #             return refine_loss
#     #         return refined_preds, mask
#     #     else:
#     #         if self.training:
#     #             voxel_features_batch = noise_rpn_features.new_ones([2,4])
#     #             coors_batch = torch.tensor([[0, 1, 2, 3], [0, 2, 2, 3]], device=voxel_features_batch.device).int()
#     #             refined_preds = self.fsaf_refiner(voxel_features_batch, coors_batch, 2, noise_rpn_features)
#     #             return refined_preds.sum()* 0
#     #         return None, None
#
#
#
#     def predict_single(self, box_preds, cls_preds, dir_preds, features, a_mask, meta, feature_map_size_prod, dtype,
#                        device, num_class_with_bg, post_center_range, predictions_dicts, points):
#         if a_mask is not None:
#             box_preds = box_preds[a_mask]
#             cls_preds = cls_preds[a_mask]
#         box_preds = box_preds.float()
#         cls_preds = cls_preds.float()
#         if self._use_direction_classifier and (dir_preds is not None):
#             if a_mask is not None:
#                 dir_preds = dir_preds[a_mask]
#             dir_labels = torch.max(dir_preds, dim=-1)[1]
#         if self.encode_background_as_zeros:
#             # this don't support softmax
#             assert self._use_sigmoid_score is True
#             total_scores = torch.sigmoid(cls_preds)
#         else:
#             # encode background as first element in one-hot vector
#             if self._use_sigmoid_score:
#                 total_scores = torch.sigmoid(cls_preds)[..., 1:]
#             else:
#                 total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
#
#         # Apply NMS in birdeye view
#         if self._use_rotate_nms:
#             nms_func = box_torch_ops.rotate_nms
#         else:
#             nms_func = box_torch_ops.nms
#         if self._multiclass_nms:
#             assert self.encode_background_as_zeros is True
#             boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
#             if not self._use_rotate_nms:
#                 box_preds_corners = box_torch_ops.center_to_corner_box2d(
#                     boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
#                     boxes_for_nms[:, 4])
#                 boxes_for_nms = box_torch_ops.corner_to_standup_nd(
#                     box_preds_corners)
#
#             selected_boxes, selected_labels, selected_scores = [], [], []
#             selected_dir_labels = []
#
#             boxes = boxes_for_nms
#             selected_per_class = []
#             score_threshs = self._nms_score_thresholds
#             pre_max_sizes = self._nms_pre_max_sizes
#             post_max_sizes = self._nms_post_max_sizes
#             iou_thresholds = self._nms_iou_thresholds
#             for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
#                     range(self.num_class),
#                     score_threshs,
#                     pre_max_sizes, post_max_sizes, iou_thresholds):
#                 if self._nms_class_agnostic:
#                     class_scores = total_scores.view(
#                         feature_map_size_prod, -1,
#                         self.num_class)[..., class_idx]
#                     class_scores = class_scores.contiguous().view(-1)
#                     class_boxes_nms = boxes.view(-1,
#                                                  boxes_for_nms.shape[-1])
#                     class_boxes = box_preds
#                     class_dir_labels = dir_labels
#                 else:
#                     anchors_range = self.target_assigner.anchors_range(class_idx)
#                     class_scores = total_scores.view(
#                         -1,
#                         self.num_class)[anchors_range[0]:anchors_range[1], class_idx]
#                     class_boxes_nms = boxes.view(-1,
#                                                  boxes_for_nms.shape[-1])[anchors_range[0]:anchors_range[1], :]
#                     class_scores = class_scores.contiguous().view(-1)
#                     class_boxes_nms = class_boxes_nms.contiguous().view(
#                         -1, boxes_for_nms.shape[-1])
#                     class_boxes = box_preds.view(-1,
#                                                  box_preds.shape[-1])[anchors_range[0]:anchors_range[1], :]
#                     class_boxes = class_boxes.contiguous().view(
#                         -1, box_preds.shape[-1])
#                     if self._use_direction_classifier and (dir_preds is not None):
#                         class_dir_labels = dir_labels.view(-1)[anchors_range[0]:anchors_range[1]]
#                         class_dir_labels = class_dir_labels.contiguous(
#                         ).view(-1)
#                 if score_thresh > 0.0:
#                     class_scores_keep = class_scores >= score_thresh
#                     if class_scores_keep.shape[0] == 0:
#                         selected_per_class.append(None)
#                         continue
#                     class_scores = class_scores[class_scores_keep]
#                 if class_scores.shape[0] != 0:
#                     if score_thresh > 0.0:
#                         class_boxes_nms = class_boxes_nms[
#                             class_scores_keep]
#                         class_boxes = class_boxes[class_scores_keep]
#                         class_dir_labels = class_dir_labels[
#                             class_scores_keep]
#                     keep = nms_func(class_boxes_nms, class_scores, pre_ms,
#                                     post_ms, iou_th)
#                     if keep.shape[0] != 0:
#                         selected_per_class.append(keep)
#                     else:
#                         selected_per_class.append(None)
#                 else:
#                     selected_per_class.append(None)
#                 selected = selected_per_class[-1]
#
#                 if selected is not None:
#                     selected_boxes.append(class_boxes[selected])
#                     selected_labels.append(
#                         torch.full([class_boxes[selected].shape[0]],
#                                    class_idx,
#                                    dtype=torch.int64,
#                                    device=box_preds.device))
#                     if self._use_direction_classifier and (dir_preds is not None):
#                         selected_dir_labels.append(
#                             class_dir_labels[selected])
#                     selected_scores.append(class_scores[selected])
#             selected_boxes = torch.cat(selected_boxes, dim=0)
#             selected_labels = torch.cat(selected_labels, dim=0)
#             selected_scores = torch.cat(selected_scores, dim=0)
#             if self._use_direction_classifier and (dir_preds is not None):
#                 selected_dir_labels = torch.cat(selected_dir_labels, dim=0)
#         else:
#             # get highest score per prediction, than apply nms
#             # to remove overlapped box.
#             if num_class_with_bg == 1:
#                 top_scores = total_scores.squeeze(-1)
#                 top_labels = torch.zeros(
#                     total_scores.shape[0],
#                     device=total_scores.device,
#                     dtype=torch.long)
#             else:
#                 top_scores, top_labels = torch.max(
#                     total_scores, dim=-1)
#             if self._nms_score_thresholds[0] > 0.0:
#                 top_scores_keep = top_scores >= self._nms_score_thresholds[0]
#                 top_scores = top_scores.masked_select(top_scores_keep)
#
#             if top_scores.shape[0] != 0:
#                 if self._nms_score_thresholds[0] > 0.0:
#                     box_preds = box_preds[top_scores_keep]
#                     if self._use_direction_classifier and (dir_preds is not None):
#                         dir_labels = dir_labels[top_scores_keep]
#                     top_labels = top_labels[top_scores_keep]
#                     if (
#                             self.fsaf == 1) and self.FSAFLoss.cfg.refinement and self.global_step >= self.FSAFLoss.cfg.refinement_step:
#                         features = features[top_scores_keep]
#                         refine_preds, mask = self.refine(points, box_preds, features)
#                         if (refine_preds is not None) and len(refine_preds) > 0:
#                             top_scores = top_scores[mask]
#                             top_labels = top_labels[mask]
#                             box_preds = box_preds[mask]
#                             refine_preds[:, 3:6] = torch.exp(refine_preds[:, 3:6])
#                             box_preds[:, 3:6] = refine_preds[:, 3:6] * box_preds[:, 3:6]
#                             refine_preds[:, :2] = refine_preds[:, :2] * torch.norm(box_preds[:, 3:5], 2, dim=1,
#                                                                                    keepdim=True)
#                             refine_preds[:, 3] = refine_preds[:, 3] * box_preds[:, 3]
#                             box_preds[:, :3] += refine_preds[:, :3]
#                             box_preds[:, 6] += refine_preds[:, 6]
#
#                 boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
#                 if not self._use_rotate_nms:
#                     box_preds_corners = box_torch_ops.center_to_corner_box2d(
#                         boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
#                         boxes_for_nms[:, 4])
#                     boxes_for_nms = box_torch_ops.corner_to_standup_nd(
#                         box_preds_corners)
#                 # the nms in 3d detection just remove overlap boxes.
#                 selected = nms_func(
#                     boxes_for_nms,
#                     top_scores,
#                     pre_max_size=self._nms_pre_max_sizes[0],
#                     post_max_size=self._nms_post_max_sizes[0],
#                     iou_threshold=self._nms_iou_thresholds[0],
#                 )
#             else:
#                 selected = []
#             # if selected is not None:
#             selected_boxes = box_preds[selected]
#             if self._use_direction_classifier and (dir_preds is not None):
#                 selected_dir_labels = dir_labels[selected]
#             selected_labels = top_labels[selected]
#             selected_scores = top_scores[selected]
#         # finally generate predictions.
#         if selected_boxes.shape[0] != 0:
#             box_preds = selected_boxes
#             scores = selected_scores
#             label_preds = selected_labels
#             if self._use_direction_classifier and (dir_preds is not None):
#                 dir_labels = selected_dir_labels
#                 period = (2 * np.pi / self._num_direction_bins)
#                 # dir_rot = box_torch_ops.limit_period(S
#                 #     box_preds[..., 6] - self._dir_offset,
#                 #     self._dir_limit_offset, period)
#                 dir_rot = box_torch_ops.limit_period(
#                     box_preds[..., 6] - self._dir_offset,
#                     0, 2 * np.pi)
#
#                 box_preds[
#                     ...,
#                     6] = dir_rot + self._dir_offset + period * dir_labels.to(
#                     box_preds.dtype)
#             final_box_preds = box_preds
#             final_scores = scores
#             final_labels = label_preds
#             if post_center_range is not None:
#                 mask = (final_box_preds[:, :3] >=
#                         post_center_range[:3]).all(1)
#                 mask &= (final_box_preds[:, :3] <=
#                          post_center_range[3:]).all(1)
#                 predictions_dict = {
#                     "box3d_lidar": final_box_preds[mask],
#                     "scores": final_scores[mask],
#                     "label_preds": label_preds[mask],
#                     "metadata": meta,
#                 }
#             else:
#                 predictions_dict = {
#                     "box3d_lidar": final_box_preds,
#                     "scores": final_scores,
#                     "label_preds": label_preds,
#                     "metadata": meta,
#                 }
#         else:
#             predictions_dict = {
#                 "box3d_lidar":
#                     torch.zeros([0, box_preds.shape[-1]],
#                                 dtype=dtype,
#                                 device=device),
#                 "scores":
#                     torch.zeros([0], dtype=dtype, device=device),
#                 "label_preds":
#                     torch.zeros([0], dtype=top_labels.dtype, device=device),
#                 "metadata":
#                     meta,
#             }
#         '''
#         for i in range(8000):
#             if not os.path.exists('npys/total_scores_%i.npy'%i):
#                 np.save('npys/total_scores_%i.npy'%i, total_scores.cpu().numpy())
#                 try:
#                     np.save('npys/final_box_preds_%i.npy'%i, final_box_preds.cpu().numpy())
#                     np.save('npys/final_scores_%i.npy'%i, final_scores.cpu().numpy())
#                     np.save('npys/label_preds_%i.npy'%i, label_preds.cpu().numpy())
#                 except:
#                     shabi=1
#                 break
#         '''
#         predictions_dicts.append(predictions_dict)
#
#     def predict(self, example, preds_dict):
#         """start with v1.6.0, this function don't contain any kitti-specific code.
#         Returns:
#             predict: list of pred_dict.
#             pred_dict: {
#                 box3d_lidar: [N, 7] 3d box.
#                 scores: [N]
#                 label_preds: [N]
#                 metadata: meta-data which contains dataset-specific information.
#                     for kitti, it contains image idx (label idx),
#                     for nuscenes, sample_token is saved in it.
#             }
#         """
#         meta_list = example["metadata"]
#         batch_size = len(meta_list)
#         t = time.time()
#         num_class_with_bg = self.num_class
#         if not self.encode_background_as_zeros:
#             num_class_with_bg = self.num_class + 1
#         predictions_dicts = []
#         post_center_range = None
#         if len(self._post_center_range) > 0:
#             post_center_range = torch.tensor(
#                 self._post_center_range,
#                 dtype=torch.float32,
#                 device=torch.cuda.current_device()).float()
#         if (self.fsaf % 2 == 0) and (self.eval_fsaf % 2 == 0):
#             batch_anchors = example["anchors"].view(batch_size, -1,
#                                                     example["anchors"].shape[-1])
#             if "anchors_mask" not in example:
#                 batch_anchors_mask = [None] * batch_size
#             else:
#                 batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
#
#             batch_box_preds = preds_dict["box_preds"]
#             batch_cls_preds = preds_dict["cls_preds"]
#             batch_box_preds = batch_box_preds.view(batch_size, -1,
#                                                    batch_box_preds.shape[-1])
#             batch_cls_preds = batch_cls_preds.view(batch_size, -1,
#                                                    num_class_with_bg)
#             batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
#                                                            batch_anchors)
#             '''
#             if self.use_iou_branch:
#                 batch_iou_preds = preds_dict["iou_preds"]
#                 batch_iou_preds = batch_iou_preds.view(batch_size, -1, 1)
#                 batch_cls_preds *= batch_iou_preds
#             '''
#             if self._use_direction_classifier:
#                 batch_dir_preds = preds_dict["dir_cls_preds"]
#                 batch_dir_preds = batch_dir_preds.view(batch_size, -1,
#                                                        self._num_direction_bins)
#             else:
#                 batch_dir_preds = [None] * batch_size
#             feature_map_size_prod = batch_box_preds.shape[
#                                         1] // self.target_assigner.num_anchors_per_location
#         if (self.fsaf > 0) and (self.eval_fsaf > 0):  # currently only support one level
#             fsaf_batch_cls_preds, fsaf_batch_box_preds = preds_dict['fsaf'][0][0], preds_dict['fsaf'][1][0]
#             points = generate_points(fsaf_batch_cls_preds.shape[-2:], self.FSAFLoss.pc_range,
#                                      fsaf_batch_cls_preds.device)
#             if self.FSAFLoss.cfg.sphere:
#                 points[:, 0], points[:, 1] = points[:, 0] * torch.cos(points[:, 1]), points[:, 0] * torch.sin(
#                     points[:, 1])
#             fsaf_batch_box_preds = fsaf_batch_box_preds.permute(0, 2, 3, 1)
#             fsaf_batch_cls_preds = fsaf_batch_cls_preds.permute(0, 2, 3, 1)
#             fsaf_batch_box_preds = fsaf_batch_box_preds.view(batch_size, -1,
#                                                              fsaf_batch_box_preds.shape[-1])
#             fsaf_batch_cls_preds = fsaf_batch_cls_preds.view(batch_size, -1,
#                                                              num_class_with_bg)
#             if self.use_iou_branch:
#                 fsaf_batch_iou_preds = preds_dict['fsaf'][2][0]
#             if self.FSAFLoss.cfg.refinement:
#                 fsaf_batch_features = preds_dict['fsaf'][4][0]
#                 fsaf_batch_features = fsaf_batch_features.permute(0, 2, 3, 1)
#                 fsaf_batch_features = fsaf_batch_features.view(batch_size, -1, fsaf_batch_features.shape[-1])
#             '''
#             if self.FSAFLoss.cfg.centerness:
#                 fsaf_batch_centerness_preds = preds_dict['fsaf'][3][0]
#                 fsaf_batch_centerness_preds = fsaf_batch_centerness_preds.permute(0, 2, 3, 1)
#                 fsaf_batch_centerness_preds = fsaf_batch_centerness_preds.view(batch_size, -1, fsaf_batch_centerness_preds.shape[-1])
#                 fsaf_batch_centerness_preds = torch.exp(-0.2*fsaf_batch_centerness_preds.norm(dim=-1, keepdim=True))
#             if fsaf_batch_iou_preds is not None:
#                 fsaf_batch_iou_preds = fsaf_batch_iou_preds.permute(0, 2, 3, 1)
#                 fsaf_batch_iou_preds = fsaf_batch_iou_preds.view(batch_size, -1, fsaf_batch_iou_preds.shape[-1])
#                 fsaf_batch_cls_preds *= fsaf_batch_iou_preds
#             '''
#             if 'center' in self.FSAFLoss.cfg.loc_type:
#                 # centers_2d = ((batch_box_preds[:, :, :2] - 0.5)*batch_box_preds[:, :, 3:5]).contiguous().view(-1, 1, 2)
#                 if self.FSAFLoss.cfg.rot_type == 'cos_sin':
#                     '''
#                     rot_mat_T = torch.stack(
#                          [tstack([batch_box_preds[:, :, 6].contiguous().view(-1), -batch_box_preds[:, :, 7].contiguous().view(-1)]),
#                           tstack([batch_box_preds[:, :, 7].contiguous().view(-1), batch_box_preds[:, :, 6].contiguous().view(-1)])])
#                     centers_2d = torch.einsum('aij,jka->aik', (centers_2d, rot_mat_T)).view(batch_size, -1, 2)
#                     '''
#                     fsaf_batch_box_preds[:, :, 6] = torch.atan2(fsaf_batch_box_preds[:, :, 7],
#                                                                 fsaf_batch_box_preds[:, :, 6])
#                     fsaf_batch_box_preds = fsaf_batch_box_preds[:, :, :7]
#                 if self.FSAFLoss.cfg.rot_type == 'softbin_cos_sin':
#                     fsaf_batch_box_preds = fsaf_batch_box_preds[:, :, :7]
#                 # else:
#                 #    centers_2d = box_torch_ops.rotation_2d(centers_2d, fsaf_batch_box_preds[:, :, -1].view(-1)).view(batch_size, -1, 2)
#                 # batch_box_preds[:, :, :2] = points[:, :2] - centers_2d
#                 fsaf_batch_box_preds[:, :, :2] += points[:, :2]
#                 # batch_box_preds[:, :, 2] = batch_box_preds[:, :, 2] * self.FSAFLoss.dims[2] + self.FSAFLoss.pc_range[2]
#                 # batch_box_preds[:, :, 2] = batch_box_preds[:, :, 2] * 2 + 0.5 * (self.FSAFLoss.pc_range[2]+self.FSAFLoss.pc_range[5])
#                 fsaf_batch_box_preds[:, :, 3:6] = torch.exp(fsaf_batch_box_preds[:, :, 3:6])
#                 if 'bottom' in self.FSAFLoss.cfg.h_loc_type:
#                     fsaf_batch_box_preds[:, :, 2] += 0.5 * fsaf_batch_box_preds[:, :, 5]
#             else:
#                 fsaf_batch_box_preds *= self.FSAFLoss.norm_factor
#                 centers_2d = 0.5 * torch.cat([-fsaf_batch_box_preds[:, :, 0:1] + fsaf_batch_box_preds[:, :, 3:4],
#                                               fsaf_batch_box_preds[:, :, 1:2] - fsaf_batch_box_preds[:, :, 4:5]],
#                                              -1).reshape(-1, 1, 2)
#                 centers_2d = box_torch_ops.rotation_2d(centers_2d, fsaf_batch_box_preds[:, :, -1].reshape(-1)).reshape(
#                     batch_size, -1, 2)
#                 centers_2d += points[:, :2]
#                 dims = fsaf_batch_box_preds[:, :, :3] + fsaf_batch_box_preds[:, :, 3:6]
#                 fsaf_batch_box_preds[:, :, :2] = centers_2d
#                 fsaf_batch_box_preds[:, :, 2] = 0.5 * (
#                             points[:, 2] - fsaf_batch_box_preds[:, :, 2] + points[:, 2] + fsaf_batch_box_preds[:, :, 5])
#                 fsaf_batch_box_preds[:, :, 3:6] = dims
#             # batch_box_preds[:, :, 6] = 0
#         if self.eval_fsaf == 1:
#             batch_anchors_mask = [None] * batch_size
#             num_anchors_per_location = 1
#             # print('box_preds', batch_box_preds)
#             batch_box_preds = fsaf_batch_box_preds
#             batch_cls_preds = fsaf_batch_cls_preds
#             '''
#             if self.FSAFLoss.cfg.centerness:
#                 batch_centerness_preds = fsaf_batch_centerness_preds
#             '''
#             batch_dir_preds = [None] * batch_size
#             feature_map_size_prod = batch_box_preds.shape[1]
#         if (self.fsaf == 2) and (self.eval_fsaf == 2):
#             batch_box_preds = torch.cat([batch_box_preds, fsaf_batch_box_preds], 1)
#             batch_cls_preds = torch.cat([batch_cls_preds, fsaf_batch_cls_preds], 1)
#             if self._use_direction_classifier:
#                 batch_dir_preds = torch.cat([batch_dir_preds, batch_dir_preds.new_zeros(
#                     fsaf_batch_cls_preds.shape[:-1] + (batch_dir_preds.shape[-1],))], 1)
#             if batch_anchors_mask[0] is not None:
#                 batch_anchors_mask = torch.cat([batch_anchors_mask,
#                                                 fsaf_batch_box_preds.new_ones(fsaf_batch_box_preds.shape[:2],
#                                                                               dtype=torch.uint8)], 1)
#         batch_features = [None] * batch_size
#         if (self.fsaf == 1) and self.FSAFLoss.cfg.refinement:
#             points = example['points'][0]
#             batch_features = fsaf_batch_features
#         else:
#             points = None
#         for box_preds, cls_preds, dir_preds, features, a_mask, meta in zip(
#                 batch_box_preds, batch_cls_preds, batch_dir_preds,
#                 batch_features, batch_anchors_mask, meta_list):
#             self.predict_single(box_preds, cls_preds, dir_preds, features, a_mask, meta, feature_map_size_prod,
#                                 batch_box_preds.dtype, batch_box_preds.device, num_class_with_bg, post_center_range,
#                                 predictions_dicts, points)
#         return predictions_dicts
#
#     def metrics_to_float(self):
#         self.rpn_acc.float()
#         self.rpn_metrics.float()
#         self.rpn_cls_loss.float()
#         self.rpn_loc_loss.float()
#         self.rpn_total_loss.float()
#
#     def update_metrics(self, cls_loss, loc_loss, cls_preds, labels, sampled):
#         batch_size = cls_preds.shape[0]
#         num_class = self.num_class
#         if not self.encode_background_as_zeros:
#             num_class += 1
#         cls_preds = cls_preds.view(batch_size, -1, num_class)
#         rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
#         prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
#         prec = prec.numpy()
#         recall = recall.numpy()
#         rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
#         rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
#         ret = {
#             "loss": {
#                 "cls_loss": float(rpn_cls_loss),
#                 "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
#                 'loc_loss': float(rpn_loc_loss),
#                 "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
#             },
#             "rpn_acc": float(rpn_acc),
#             "pr": {},
#         }
#         for i, thresh in enumerate(self.rpn_metrics.thresholds):
#             ret["pr"][f"prec@{int(thresh * 100)}"] = float(prec[i])
#             ret["pr"][f"rec@{int(thresh * 100)}"] = float(recall[i])
#         return ret
#
#     def clear_metrics(self):
#         self.rpn_acc.clear()
#         self.rpn_metrics.clear()
#         self.rpn_cls_loss.clear()
#         self.rpn_loc_loss.clear()
#         self.rpn_total_loss.clear()
#
#     @staticmethod
#     def convert_norm_to_float(net):
#         '''
#         BatchNorm layers to have parameters in single precision.
#         Find all layers and convert them back to float. This can't
#         be done with built in .apply as that function will apply
#         fn to all modules, parameters, and buffers. Thus we wouldn't
#         be able to guard the float conversion based on the module type.
#         '''
#         if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
#             net.float()
#         for child in net.children():
#             VoxelNet.convert_norm_to_float(child)
#         return net


