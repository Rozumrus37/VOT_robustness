from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import time
import sys
import torch
import random
import numpy as np
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
seed_torch(1000000007)
torch.set_num_threads(12)



#==========vot
import pytracking.evaluation.vot2020 as vot
from pytracking.vot20_utils import *
#==========vot

#==========mask
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.VOT2021.transt_cvt_seg_class import TRANST_CVT_SEG
#==========mask


#==========base_tracker
from lib.test.tracker.mixformer_online_dimp_reppoint import MixFormerOnline
import lib.test.parameter.mixformer_online_dimprobust_reppoint_deep_dimp8 as vot_params
#==========base_tracker

class DAMTMask(object):
    def __init__(self, param, base_tracker_model, mask_model):
        params = vot_params.parameters(param, model=base_tracker_model,mask_model = mask_model, search_area_scale = 6)
        self.base_tracker = MixFormerOnline(params)
        net = NetWithBackbone(net_path=params.mask_model,use_gpu=True)
        self.mask_net = TRANST_CVT_SEG(net=net,mask_threshold=params.mask_threshold,exemplar_size=params.mask_exemplar_size, instance_size=params.mask_instance_size)
        # self.mask_net = TRANST_CVT_SEG(net=net,mask_threshold=0.60,exemplar_size=256, instance_size=256)

    def initialize(self, image, mask):
        region = rect_from_mask(mask)
        init_info = {'init_bbox': region}
        self.base_tracker.initialize(image, init_info, use_motion_strategy = False)
        self.mask_net.initialize(image, mask)

    def track(self, img_RGB):
        '''base tracker'''
        outputs = self.base_tracker.track(img_RGB.copy())
        pred_bbox = np.array(outputs['target_bbox']).copy()
        '''base tracker'''
        final_mask = self.mask_net.track(img_RGB.copy(),pred_bbox)
        return final_mask, 1

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)



param = 'baseline_large'
base_tracker_model = "mixformer_online_dimprobust_reppoint_deep_alldata.pth.tar"
mask_model = "TransTCVTLargeiouhsegm_ep0120.pth.tar"

tracker = DAMTMask(param, base_tracker_model, mask_model)
handle = vot.VOT("mask")
selection = handle.region()
imagefile = handle.frame()

if not imagefile:
    sys.exit(0)

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
mask = make_full_size(selection, (image.shape[1], image.shape[0]))

tracker.H = image.shape[0]
tracker.W = image.shape[1]

tracker.initialize(image, mask)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    region, confidence = tracker.track(image)
    handle.report(region, confidence)
