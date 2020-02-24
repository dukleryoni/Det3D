import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer



@NECKS.register_module
class RPNNoHeadBase(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 ohs=None,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn',
                 im_num_filters=None,
                 fsaf=0,
                 logger=None,
                 fsaf_cfg=None,
                 use_iou_branch=False):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self._num_out_filters = num_out_filters
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self, x):
        ups = []
        stage_outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))

        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x
        return res

@NECKS.register_module
class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 ohs=None,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn',
                 im_num_filters=None,
                 fsaf=0,
                 logger=None,
                 fsaf_cfg=None,
                 use_iou_branch=False): # can remove rpn use iou, im_num_filters, box_code_size maybe
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """

        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=ohs.encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name,
            im_num_filters=im_num_filters,
        )

        # super(RPNBase, self).__init__(
        #     use_norm=use_norm,
        #     num_class=num_class,
        #     layer_nums=layer_nums,
        #     layer_strides=layer_strides,
        #     num_filters=num_filters,
        #     upsample_strides=upsample_strides,
        #     num_upsample_filters=num_upsample_filters,
        #     num_input_features=num_input_features,
        #     num_anchor_per_loc=num_anchor_per_loc,
        #     encode_background_as_zeros=encode_background_as_zeros,
        #     use_direction_classifier=use_direction_classifier,
        #     use_groupnorm=use_groupnorm,
        #     num_groups=num_groups,
        #     box_code_size=box_code_size,
        #     num_direction_bins=num_direction_bins,
        #     name=name,
        #     im_num_filters=im_num_filters,
        #     fsaf=fsaf,
        #     fsaf_cfg=fsaf_cfg,
        #     logger=logger,
        #     use_iou_branch=use_iou_branch)


        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        self.fsaf = fsaf
        self.fsaf_cfg = fsaf_cfg
        self.use_iou_branch = use_iou_branch

        if ohs is not None:
            self.ohs = ohs
            self.fsaf = ohs.fsaf
            self.fsaf_cfg = ohs.fsaf_module
            self.use_iou_branch = ohs.use_iou_branch
            self.num_class = len(ohs.tasks)

        self._num_class = self.num_class




        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        # final_num_filters = 128
        if self.fsaf > 0:
            assert self.fsaf_cfg.loc_type in ['center', 'part',
                                              'center_softbin'], "loc_type must be one of [center, part, center_softbin]"
            assert self.fsaf_cfg.rot_type in ['direct', 'softbin', 'cos_sin',
                                              'softbin_cos_sin'], "rot_type must be one of [direct, softbin, cos_sin, softbin_cos_sin]"
            assert self.fsaf_cfg.dim_type in ['norm', 'log'], "dim_type must be one of [norm, log]"
            assert self.fsaf_cfg.h_loc_type in ['direct', 'norm', 'standard', 'softbin',
                                                'bottom_softbin'], "h_loc_type must be one of [direct, norm, standard, softbin]"

            BatchNorm2d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            self.new_fsaf_reg_conv = nn.Sequential(
                nn.Conv2d(final_num_filters, final_num_filters, kernel_size=3, padding=1),
                # DeformConv2d(final_num_filters, final_num_filters, 3, padding=1, modulation=True),
                BatchNorm2d(final_num_filters),
                nn.ReLU(),
            )

            loc_num_filters = final_num_filters

            if self.fsaf_cfg.rot_type == 'direct':
                self.new_fsaf_rot = nn.Conv2d(final_num_filters, 1, kernel_size=1)
            if 'softbin' in self.fsaf_cfg.rot_type:
                self.rot_bins = torch.linspace(-np.pi, np.pi, self.fsaf_cfg.rot_bin_num).reshape(1,
                                                                                                 self.fsaf_cfg.rot_bin_num,
                                                                                                 1, 1)
                self.new_fsaf_rot = nn.Sequential(
                    nn.Conv2d(final_num_filters, self.fsaf_cfg.rot_bin_num, kernel_size=1),
                    nn.Softmax(dim=1),
                )

            if self.fsaf_cfg.rot_type == 'cos_sin':
                self.new_fsaf_rot = nn.Sequential(
                    nn.Conv2d(final_num_filters, 2, kernel_size=1),
                )

            if self.fsaf_cfg.rot_type == 'softbin_cos_sin':
                self.new_fsaf_cos_sin = nn.Conv2d(final_num_filters, 2, kernel_size=1)
            if self.fsaf_cfg.loc_type == 'center':
                self.new_fsaf_loc = nn.Conv2d(loc_num_filters, 2, kernel_size=1)
            elif self.fsaf_cfg.loc_type == 'center_softbin':
                self.loc_bins = torch.linspace(-self.fsaf_cfg.center_range, self.fsaf_cfg.center_range,
                                               self.fsaf_cfg.center_bin_num).reshape(1, 1, -1, 1, 1)
                self.new_fsaf_loc = nn.Sequential(
                    # nn.Conv2d(loc_num_filters, 2 * self.fsaf_cfg.center_bin_num, kernel_size=1),
                    nn.Conv2d(loc_num_filters, loc_num_filters // 2, kernel_size=1),
                    BatchNorm2d(loc_num_filters // 2),
                    nn.ReLU(),
                    nn.Conv2d(loc_num_filters // 2, 2 * self.fsaf_cfg.center_bin_num, kernel_size=1),
                    # nn.Softmax(dim=1),
                )
            else:
                self.new_fsaf_loc = nn.Sequential(
                    nn.Conv2d(loc_num_filters, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
            if self.fsaf_cfg.centerness:
                self.fsaf_cfg.part_classification = False
                self.new_fsaf_centerness = nn.Conv2d(final_num_filters, 2, kernel_size=1)
            if 'softbin' in self.fsaf_cfg.h_loc_type:
                self.h_loc_bins = torch.linspace(-3, 1, self.fsaf_cfg.h_bin_num).reshape(1, self.fsaf_cfg.h_bin_num, 1,
                                                                                         1)
                self.new_fsaf_h = nn.Sequential(
                    nn.Conv2d(final_num_filters, self.fsaf_cfg.h_bin_num, kernel_size=1),
                    nn.Softmax(dim=1),
                )
            else:
                self.new_fsaf_h = nn.Conv2d(final_num_filters, 1, kernel_size=1)
            self.new_fsaf_dim = nn.Conv2d(final_num_filters, 3, kernel_size=1)
            self.new_fsaf_cls = nn.Conv2d(final_num_filters, self._num_class, 1)
            if self.use_iou_branch:
                if self.fsaf_cfg.rot_type == 'cos_sin':
                    box_code_num = 8
                else:
                    box_code_num = 7
                self.new_fsaf_iou_branch = nn.Sequential(
                    nn.Conv2d(final_num_filters + box_code_num, int(final_num_filters / 2), kernel_size=3, padding=1),
                    BatchNorm2d(int(final_num_filters / 2)),
                    nn.ReLU(),
                    nn.Conv2d(int(final_num_filters / 2), 1, 1),
                    nn.Sigmoid(),
                )
            if self.fsaf_cfg.part_classification:
                if self.fsaf_cfg.part_type in ['up_down', 'left_right']:
                    part_num = 2
                elif self.fsaf_cfg.part_type == 'quadrant':
                    part_num = 4
                else:
                    part_num = 8
                self.new_fsaf_part_branch = nn.Sequential(
                    nn.Conv2d(final_num_filters, part_num, 1),
                    nn.Sigmoid(),
                )

            if use_direction_classifier:
                if im_num_filters is not None:
                    self.new_conv_dir_cls = nn.Conv2d(
                        final_num_filters, self._num_anchor_per_loc * num_direction_bins, 1)
                else:
                    self.conv_dir_cls = nn.Conv2d(
                        final_num_filters, self._num_anchor_per_loc * num_direction_bins, 1)

            # self.new_conv_fuse_dir = nn.Conv2d(final_num_filters+im_num_filters, final_num_filters, 1)

    #    logger.info("Finish RPN_OHS Initialization")

    def fsaf_forward(self, x):
        x = self.new_fsaf_reg_conv(x)
        cls_score = self.new_fsaf_cls(x)
        # x = torch.cat([x, occupation], 1)
        rot = self.new_fsaf_rot(x)
        if 'softbin' in self.fsaf_cfg.rot_type:
            rot = rot * self.rot_bins.to(x.device)
            rot = torch.sum(rot, dim=1, keepdim=True)
        if self.fsaf_cfg.rot_type == 'cos_sin':
            rot = F.normalize(rot, p=2, dim=1)
        # if self.fsaf_cfg.rot_type == 'softbin_cos_sin':
        #     cos_sin = self.new_fsaf_cos_sin(x)
        #     cos_sin = F.normalize(cos_sin, p=2, dim=1)
        #     rot = torch.cat([rot, cos_sin], 1)
        dim = self.new_fsaf_dim(x)
        if not self.fsaf_cfg.dim_type == 'log':
            dim = F.relu(dim)
        if self.fsaf_cfg.part_classification:
            part = self.new_fsaf_part_branch(x)
        elif self.fsaf_cfg.centerness:
            part = self.new_fsaf_centerness(x)
        else:
            part = None
        h = self.new_fsaf_h(x)
        if 'softbin' in self.fsaf_cfg.h_loc_type:
            h = h * self.h_loc_bins.to(x.device)
            h = torch.sum(h, dim=1, keepdim=True)
        loc = self.new_fsaf_loc(x)
        if self.fsaf_cfg.loc_type == 'center_softbin':
            loc = loc.view((loc.shape[0], 2, self.fsaf_cfg.center_bin_num, loc.shape[2], loc.shape[3]))
            loc = F.softmax(loc, dim=2)
            loc = loc * self.loc_bins.to(x.device)
            loc = torch.sum(loc, dim=2, keepdim=False)
        bbox_pred = torch.cat([loc, h, dim, rot], 1)
        if self.use_iou_branch:
            iou_pred = self.new_fsaf_iou_branch(torch.cat([x, bbox_pred], 1))
        else:
            iou_pred = None
        if self.fsaf_cfg.refinement:
            return [cls_score], [bbox_pred], [iou_pred], [part], [x]
        return [cls_score], [bbox_pred], [iou_pred], [part], [None]

    def forward(self, x, features2d=None,use_head=False):
        ret_dict = dict()
        if len(x.shape) == 5:
            _N, _C, _D, _H, _W = x.shape
            res = super().forward(x.view(_N, _C * _D, _H, _W))
        else:
            res = super().forward(x)
        out = res["out"]
        ret_dict["out"] = out
        # out = x.view(_N, _C*_D, _H, _W)
        if self.fsaf > 0:
            # out_fsaf = self.fsaf_forward_center(torch.cat([out.unsqueeze(2).repeat(1,1,_D, 1, 1), x], 1))
            # out_fsaf = self.fsaf_forward(out)
            out_fsaf = self.fsaf_forward(out.clone())
            ret_dict['fsaf'] = out_fsaf

        if use_head:
            return out

        return ret_dict


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

@NECKS.register_module
class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            # if self._use_groupnorm:
            #     BatchNorm2d = change_default_args(
            #         num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            # else:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)

            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes


