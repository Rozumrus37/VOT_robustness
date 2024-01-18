#import packages
from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import os
import sys
import numpy as np
from vot.region import io

seq_name, sub_dir = sys.argv[1], sys.argv[2]
setup_num, sq_len, thick, pix_cov, seq_name =  sys.argv[1], sys.argv[2],  sys.argv[3], sys.argv[4], sys.argv[5]


def apply_mask_to_image(image, binary_mask, offset):
    mask_indices = np.argwhere(binary_mask == 1)

    for index in mask_indices:
        x, y = index 
        x += offset[1]
        y += offset[0]
        if 0 < y <= image.width and 0 < x <= image.height:
            image.putpixel((y-1, x-1), (255, 192, 203))


def composite_images(img1, img2):
    width = img1.width() + img2.width()
    height = max(img1.height(), img2.height())


    composite_img = Image.new("RGB", (width, height), "white")
    composite_img.paste(img1, (0, 0, img1.width(), img1.height()))
    composite_img.paste(img2, (img1.width(), 0, img1.width() + img2.width(), img2.height()))
    tk_img = ImageTk.PhotoImage(composite_img)

    return tk_img


root = tk.Tk()
root.title("Prediction ---- Groundtruth DAMTMask")
root.resizable()
frame = tk.Frame(root)
frame.pack(pady=10)


imgs_pred = []

path = os.path.join("results/DAMTMask/baseline", "setup_" + seq_name + "_" + setup_num, sq_len + "_" + thick + "_" + pix_cov + "_color")  

imgs = []
for filename in os.listdir(path):
    if filename[0] != ".":
        imgs.append(filename)
imgs.sort()

file_path = os.path.join("results/DAMTMask/baseline", "setup_" + seq_name + "_" + setup_num, sq_len + "_" + thick + "_" + pix_cov + "/" + seq_name + "_00000000.txt")   
file_path_gt = os.path.join("results/DAMTMask/baseline", "setup_" + seq_name + "_" + setup_num, sq_len + "_" + thick + "_" + pix_cov + "_gt.txt")  


with open(file_path_gt, 'r') as file:
    lines_gt = file.readlines()

with open(file_path, 'r') as file:
    lines_pred = file.readlines()


strings_pred = [line.strip() for line in lines_pred[1:]]


strings_pred = [lines_gt[0].strip()] + strings_pred
strings_gt = [line.strip() for line in lines_gt]


im_curr = None

for i in range(len(strings_pred)):
    filename = imgs[i]
    s = strings_pred[i]
    img1 = Image.open(os.path.join(path, filename))
    binary_mask = np.array(io.parse_region(s)._mask)
    offset = io.parse_region(s)._offset
    apply_mask_to_image(img1, binary_mask, offset)

    filename = imgs[i]
    s = strings_gt[i]
    img2 = Image.open(os.path.join(path, filename))
    binary_mask = np.array(io.parse_region(s)._mask)
    offset = io.parse_region(s)._offset
    apply_mask_to_image(img2, binary_mask, offset)

    images = [img1, img2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    if i == 2:
        im_curr = img2

    newsize = (int(total_width/1), int(max_height/1))
    new_im = new_im.resize(newsize)

    imgs_pred.append(ImageTk.PhotoImage(new_im))


List_img = imgs_pred

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



