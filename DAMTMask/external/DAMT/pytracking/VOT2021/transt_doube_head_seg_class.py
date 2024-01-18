from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time
import os
from pytracking.vot20_utils import *
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
import pytracking.evaluation.vot2020 as vot
''''''
class TRANST_RES_SEG(object):

    def __init__(self, net, mask_threshold=0.5,
                 exemplar_size=256, instance_size=256):
        self.net = net
        self.mask_threshold = mask_threshold
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size


    def _convert_mask(self, delta):
        delta = delta.squeeze(0).squeeze(0)
        delta = delta.data.cpu().numpy()
        return delta

    def get_subwindow_with_mask(self, im,mask, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        mask = mask[:,:,None].astype(np.float32)
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        _,_,k_mask = mask.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im

            size_mask = (r + top_pad + bottom_pad, c + left_pad + right_pad, k_mask)
            te_mask = np.zeros(size_mask, np.float32)
            te_mask[top_pad:top_pad + r, left_pad:left_pad + c, :] = mask

            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
                te_mask[0:top_pad, left_pad:left_pad + c, :] = 0
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
                te_mask[r + top_pad:, left_pad:left_pad + c, :] = 0
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
                te_mask[:, 0:left_pad, :] = 0
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
                te_mask[:, c + left_pad:, :] = 0
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
            mask_patch = te_mask[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
            mask_patch = mask[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
            mask_patch = cv2.resize(mask_patch, (model_sz, model_sz))
            if len(mask_patch.shape) == 2:
                mask_patch = mask_patch[:,:,None]

        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()

        mask_patch = mask_patch.transpose(2, 0, 1)
        mask_patch = mask_patch[np.newaxis, :, :, :]
        mask_patch = mask_patch.astype(np.float32)
        mask_patch = torch.from_numpy(mask_patch)
        mask_patch = mask_patch.cuda()
        return im_patch,mask_patch

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def get_subwindow_numpy(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.float32)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))

        return im_patch

    def map_mask_back(self, im, center_pos, instance_size, s_x, mask, mode=cv2.BORDER_REPLICATE):
        """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        H, W = (im.shape[0], im.shape[1])
        base = np.zeros((H, W))
        x_center, y_center = center_pos.tolist()

        # Crop image

        if s_x < 1 or s_x < 1:
            raise Exception('Too small bounding box.')
        c = (s_x + 1) / 2

        x1 = int(np.floor(x_center - c + 0.5))
        x2 = int(x1 + s_x - 1)

        y1 = int(np.floor(y_center - c + 0.5))
        y2 = int(y1 + s_x -1)

        x1_pad = int(max(0., -x1))
        y1_pad = int(max(0., -y1))
        x2_pad = int(max(0., x2 - W + 1))
        y2_pad = int(max(0., y2 - H + 1))

        '''pad base'''
        base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
        '''Resize mask'''
        mask_rsz = cv2.resize(mask, (s_x, s_x))
        '''fill region with mask'''
        base_padded[y1 + y1_pad:y2 + y1_pad + 1, x1 + x1_pad:x2 + x1_pad + 1] = mask_rsz.copy()
        '''crop base_padded to get final mask'''
        final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
        assert (final_mask.shape == (H, W))
        return final_mask

    def constraint_mask(self, mask, bbox):
        """
        mask: shape (H, W)
        bbox: list [x1, y1, w, h]
        """
        x1 = np.int(np.floor(bbox[0]))
        y1 = np.int(np.floor(bbox[1]))
        x2 = np.int(np.ceil(bbox[0]+bbox[2]))
        y2 = np.int(np.ceil(bbox[1]+bbox[3]))
        mask[0:y1+1,:] = 0
        mask[y2:,:] = 0
        mask[:,0:x1+1] = 0
        mask[:,x2:] = 0
        if mask.max() == 0:
            yp1 = np.int(np.floor(bbox[1]+bbox[3]/4))
            yp2 = np.int(np.ceil(bbox[1]+3*bbox[3]/4))
            xp1 = np.int(np.floor(bbox[0]+bbox[2]/4))
            xp2 = np.int(np.ceil(bbox[0]+3*bbox[2]/4))
            mask[yp1:yp2,xp1:xp2] = 1
        return mask

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, mask):
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()

        region = rect_from_mask(mask)
        gt_bbox_np = np.array(region).astype(np.float32)
        bbox = gt_bbox_np

        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop,z_mask = self.get_subwindow_with_mask(image,mask.astype(np.float32), self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        z_crop = torch.stack([z_crop,z_crop],1)
        z_mask = torch.stack([z_mask,z_mask],1)
        template = self.net.get_template(z_crop,z_mask)
        self.template = template

    def track(self, image, bbox):
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])
        # calculate x crop size
        w_x = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop_ori = self.get_subwindow(image, self.center_pos,
                                        self.instance_size,
                                        round(s_x), self.channel_average)
        #==========show
        self.vis_img_transt = x_crop_ori.squeeze(0).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)
        self.vis_center_transt = self.center_pos.copy()
        self.vis_size_transt = s_x
        #==========show
        # normalize
        x_crop = x_crop_ori.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)

        # track
        outputs = self.net.get_mask(x_crop, self.template)

        pred_mask_search_transformer = self._convert_mask(outputs['pred_masks_transformer'])
        self.vis_transtcvt_maskraw_transformer = cv2.resize(pred_mask_search_transformer.copy(),(self.instance_size,self.instance_size))

        pred_mask_search_cnn = self._convert_mask(outputs['pred_masks_cnn'])
        self.vis_transtcvt_maskraw_cnn = cv2.resize(pred_mask_search_cnn.copy(),(self.instance_size,self.instance_size))

        pred_mask_transformer = self.map_mask_back(image, self.center_pos, self.instance_size, s_x, pred_mask_search_transformer,mode=cv2.BORDER_CONSTANT)
        pred_mask_cnn = self.map_mask_back(image, self.center_pos, self.instance_size, s_x, pred_mask_search_cnn,mode=cv2.BORDER_CONSTANT)


        self.final_mask_raw_transformer = pred_mask_transformer
        self.final_mask_raw_cnn = pred_mask_cnn

        final_mask_transformer = (pred_mask_transformer > self.mask_threshold).astype(np.uint8)
        final_mask_cnn = (pred_mask_cnn > self.mask_threshold).astype(np.uint8)
        # final_mask = self.constraint_mask(final_mask, bbox)

        return {'final_mask_transformer':final_mask_transformer,
                'final_mask_cnn':final_mask_cnn
                }

    def update(self, image, bbox):
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # while(True):
        #     cv2.imshow('image', image)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # calculate z crop size
        w_z = self.size[0] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (2 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)

        # initialize template feature
        template = self.net.template(z_crop)
        self.templates_list.pop(1)
        self.templates_list.append(template)
