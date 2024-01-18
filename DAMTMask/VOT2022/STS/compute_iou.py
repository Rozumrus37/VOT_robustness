#import packages
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import sys
import numpy as np
from vot.region import io

# seq_name, sub_dir = sys.argv[1], sys.argv[2]
# setup_num, sq_len, thick, pix_cov, seq_name =  sys.argv[1], sys.argv[2],  sys.argv[3], sys.argv[4], sys.argv[5]
# seq_name = "book"

def apply_mask_to_image(binary_mask, offset, seq_name):
    image = Image.open(os.path.join("sequences", seq_name, "color/00000001.jpg"))
    mask_indices = np.argwhere(binary_mask == 1)

    pixs = []
    for index in mask_indices:
        x, y = index 
        x += offset[1]
        y += offset[0]
        if 0 < y <= image.width and 0 < x <= image.height:
            pixs.append((y-1, x-1))

    return pixs


def get_iou(setup_num, sq_len, thick, pix_cov, seq_name):
    file_path = os.path.join("/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/results/DAMTMask/baseline", "setup_" + seq_name + "_" + setup_num, sq_len + "_" + thick + "_" + pix_cov + "/" + seq_name + "_00000000.txt")   
    file_path_gt = os.path.join("/home.stud/rozumrus/VOT/DAMTMask/VOT2022/STS/results/DAMTMask/baseline", "setup_" + seq_name + "_" + setup_num, sq_len + "_" + thick + "_" + pix_cov + "_gt.txt")  

    with open(file_path_gt, 'r') as file:
        lines_gt = file.readlines()

    with open(file_path, 'r') as file:
        lines_pred = file.readlines()

    def calculate_iou(pred_mask, true_mask):
        pred_mask = set(pred_mask)
        true_mask = set(true_mask)

        intersection = pred_mask & true_mask
        union = pred_mask | true_mask

        if len(union) == 0:
            return 0

        iou = len(intersection) / len(union)

        return iou


    strings_pred = [line.strip() for line in lines_pred[1:]]
    strings_gt = [line.strip() for line in lines_gt[1:]]

    iou = []

    for i in range(len(strings_pred)):
        s = strings_pred[i]
        binary_mask = np.array(io.parse_region(s)._mask)
        offset = io.parse_region(s)._offset
        pixs1 = apply_mask_to_image(binary_mask, offset, seq_name)

        s = strings_gt[i]
        binary_mask = np.array(io.parse_region(s)._mask)
        offset = io.parse_region(s)._offset
        pixs2 = apply_mask_to_image(binary_mask, offset, seq_name)

        iou.append(calculate_iou(pixs1, pixs2))

    iou = np.array(iou)
    return iou.mean() * 100
