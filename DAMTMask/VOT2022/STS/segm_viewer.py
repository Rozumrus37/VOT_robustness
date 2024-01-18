#import packages
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import sys
import numpy as np
from vot.region import io

seq_name, sub_dir = sys.argv[1], sys.argv[2]
setup_num, sq_len, thick, pix_cov =  sys.argv[1], sys.argv[2],  sys.argv[3], sys.argv[4]

def apply_mask_to_image(image, binary_mask, offset):
    mask_indices = np.argwhere(binary_mask == 1)

    for index in mask_indices:
        x, y = index 
        x += offset[1]
        y += offset[0]
        if 0 < y <= image.width and 0 < x <= image.height:
            image.putpixel((y-1, x-1), (0, 0, 0))  


root = tk.Tk()
root.title("Segm Mask viewer")
root.resizable()
frame = tk.Frame(root)
frame.pack(pady=10)


List_img = []

path = os.path.join("sequences", seq_name, sub_dir)

imgs = []

for filename in os.listdir(path):
    if filename[0] != ".":
        imgs.append(filename)

imgs.sort()


file_path = os.path.join("results/MS_AOT/baseline", "1FR_GRID_REST_GRID_2/80_43_70_init", seq_name + "_00000000.txt")  
file_path_gt = os.path.join("sequences",  seq_name, "groundtruth.txt")  
file_path = file_path_gt
with open(file_path_gt, 'r') as file:
    lines_gt = file.readlines()

with open(file_path, 'r') as file:
    lines = file.readlines()

strings = [line.strip() for line in lines[1:]]
strings = [lines_gt[0].strip()] + strings

for i in range(len(strings)):

    filename = imgs[i]

    s = strings[i]
    img = Image.open(os.path.join(path, filename))
    binary_mask = np.array(io.parse_region(s)._mask)

    offset = io.parse_region(s)._offset
    apply_mask_to_image(img, binary_mask, offset)
    List_img.append(ImageTk.PhotoImage(img))

j = 0
img_label = Label(frame, image=List_img[j])
img_label.pack()

def next_img():
   global j
   j = j + 1
   try:
       img_label.config(image=List_img[j])
   except:
       j = -1
       next_img()

def prev():
   global j
   j = j - 1
   try:
       img_label.config(image=List_img[j])
   except:
       j = 0
       prev()


frame1 = tk.Frame(root)
frame1.pack(pady=5)
Prev = tk.Button(frame1, text="Previous", command=prev)
Prev.pack(side="left", padx=10)
Next = tk.Button(frame1, text="Next", command=next_img)
Next.pack(side="right", padx=10)


frame2 = tk.Frame(root)
frame2.pack(pady=5)
Exit = tk.Button(frame2, text="Exit", command=root.quit)
Exit.pack(side="bottom")


root.mainloop()

