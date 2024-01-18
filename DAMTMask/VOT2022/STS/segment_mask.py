#import packages
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from PIL import Image, ImageDraw
import os
import sys
import numpy as np
from vot.region import io

seq_name, len_side, line_thick = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def apply_mask(width, height, binary_mask, offset):
    arr = np.zeros((height, width))
    print(height, width)

    mask_indices = np.argwhere(binary_mask == 1)

    for index in mask_indices:
        x, y = index 
        x += offset[1]
        y += offset[0]

        if 0 < y <= width and 0 < x <= height:
            arr[x-1, y-1] = 1

    return arr


def get_grid_img(width, height, lineThickness, x, lineColor=(0, 0, 0)):
    num_squares_x, num_squares_y = width // x, height // x

    total = width*height

    black_pixel_count = 0
    rgb_image = Image.new('RGB', (width, height), (255, 255, 255))
    
    draw = ImageDraw.Draw(rgb_image)
    grid_pixels = 0

    for i in range(1, num_squares_x):
        draw.line([(x * i, 0), (x * i, height)], fill=lineColor, width=lineThickness)
        grid_pixels += lineThickness * height

    for i in range(1, num_squares_y):
        draw.line([(0, x * i), (width, x * i)], fill=lineColor, width=lineThickness)
        grid_pixels += lineThickness * height

    return rgb_image.convert('L')


def segment_one(binary_mask, offset):
    img = Image.open(os.path.join("sequences", seq_name, "init_color/00000001.jpg"))
    w = img.width
    h = img.height

    arr = np.array(apply_mask(w, h, binary_mask, offset), dtype=int)

    arr_grid = np.array(np.logical_and(arr, np.array(get_grid_img(w, h, line_thick, len_side))), dtype=int)

    print(arr, arr_grid, np.sum(arr), np.sum(arr_grid))

    (tl_x, tl_y, region_w, region_h), rle = io.encode_mask(arr_grid)
    return "m" + str(tl_x) +  "," + str(tl_y) + "," + str(region_w) + "," + str(region_h) + "," +  ",".join(map(str, rle))


file_path_gt = os.path.join("sequences",  seq_name, "groundtruth.txt")  

with open(file_path_gt, 'r') as file:
    lines_gt = file.readlines()

s = lines_gt[0].strip()

binary_mask = np.array(io.parse_region(s)._mask)
offset = io.parse_region(s)._offset

print(segment_one(binary_mask, offset))
