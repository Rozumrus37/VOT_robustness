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

#==========base_tracker
from lib.test.tracker.mixformer_online_dimp_reppoint import MixFormerOnline
import lib.test.parameter.mixformer_online_dimprobust_reppoint_deep_dimp8 as vot_params
#==========base_tracker

class DAMT(object):
    def __init__(self, param, base_tracker_model):
        params = vot_params.parameters(param, model=base_tracker_model, search_area_scale = 6)
        self.base_tracker = MixFormerOnline(params)

    def initialize(self, image, init_box):
        region = np.array(init_box)
        init_info = {'init_bbox': region}
        self.base_tracker.initialize(image, init_info, use_motion_strategy = False)

    def track(self, img_RGB):
        outputs = self.base_tracker.track(img_RGB.copy())
        pred_bbox = np.array(outputs['target_bbox']).copy()
        return pred_bbox, None


param = 'baseline_large'
base_tracker_model = "mixformer_online_dimprobust_reppoint_deep_alldata.pth.tar"

tracker = DAMT(param, base_tracker_model)
handle = vot.VOT("rectangle")
selection = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
tracker.H = image.shape[0]
tracker.W = image.shape[1]
init_box = [selection.x, selection.y, selection.width, selection.height]
tracker.initialize(image, init_box)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    region, confidence = tracker.track(image)
    handle.report(vot.Rectangle(*region), confidence)
