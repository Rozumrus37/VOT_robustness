import os
import sys
import cv2
import vot
from vot_path import base_path
if base_path not in sys.path:
    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, 'utils'))
print("2")
from vitkt_m import vitTrack_Tracker
print("3")
import numpy as np
print("4")
class p_config(object):
    score_thrs=0.7
    update_score=0.8
    ctdis=0.55
# test DiMPMU
print("5")
p = p_config()
print("6")
handle = vot.VOT("rectangle")
print("7")
selection = handle.region()
imagefile = handle.frame()
print("8")

image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
print("10")
tracker = vitTrack_Tracker(image, selection, p=p)
print("9")

print("yes")
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    print("processed: ", imagefile)
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
    region, score,ap_dis,lof_dis= tracker.tracking(image)
    handle.report(vot.Rectangle(float(region[0]), float(region[1]), float(region[2]),
                float(region[3])),score)
