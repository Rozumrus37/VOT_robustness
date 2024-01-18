from PIL import Image, ImageDraw
import os
import sys
from tqdm import tqdm

def create_grid(x=40, lineColor=(0, 0, 0), lineThickness=1):  

    seqs = ['agility', 'animal', 'ants1', 'ball2', 'ball3', 'basketball', 'birds1', 'birds2'] #['airplane', 'bag', 'bicycle', 'car', 'dancingshoe', 'diving', 'goldfish']
    base_path = "/datagrid/personal/rozumrus/vot2022/sts/workspace/sequences"

    for dir_path in os.listdir(base_path):
        path_dir = "2l_1th_70p_color"

        if not dir_path in seqs:
            continue

        if dir_path[0] == ".":
            continue
        if dir_path == "grid_gen.py" or dir_path == "old_list.txt" or dir_path == "list.txt":
            continue

        if os.path.exists(os.path.join(base_path, dir_path, path_dir)):
            os.rmdir(os.path.join(base_path, dir_path, path_dir))
        
        os.mkdir(os.path.join(base_path, dir_path, path_dir))
        print(dir_path)

        for filename_in in tqdm(os.listdir(os.path.join(base_path, dir_path, "init_color"))):
            if filename_in[0] == '.':
                continue

            # print(dir_path, filename_in)

            savepath = os.path.join(base_path, dir_path, path_dir, filename_in)

            im = Image.open(os.path.join(base_path, dir_path, "init_color", filename_in))
            
            width, height = im.size[0], im.size[1]
            
            num_squares_x, num_squares_y = width // x, height // x

            draw = ImageDraw.Draw(im)
            grid_pixels = 0

            for i in range(1, num_squares_x+1):
                draw.line([(x * i, 0), (x * i, height)], fill=lineColor, width=lineThickness)
                grid_pixels += lineThickness * height

            for i in range(1, num_squares_y+1):
                draw.line([(0, x * i), (width, x * i)], fill=lineColor, width=lineThickness)
                grid_pixels += lineThickness * height
        
            im.save(savepath, "JPEG")

len_side, line_thick = sys.argv[1],  sys.argv[2]
create_grid(x=int(len_side), lineThickness=int(line_thick))


#80, 1 --> 0.022699652777777777
#8l_1th_23.2p_color
#80l_11th_23.5p_color

#2l_1th_74.8p_color
#80l_44th_75.2p_color

#80l_44th_75.2p_color