import os
import shutil



# for dir_path in os.listdir("/datagrid/personal/rozumrus/vot2022/lt/sequences"):

# 	if dir_path[0] == ".":
# 		continue
# 	if dir_path == "grid_gen.py" or dir_path == "old_list.txt" or dir_path == "list.txt":
# 	    continue

# 	if not os.path.exists(os.path.join(dir_path, "color_grid_thick")):
# 		continue

# 	shutil.move(os.path.join(dir_path, "color"), os.path.join(dir_path, "color_init1"))
# 	shutil.move(os.path.join(dir_path, "color_grid_thick"), os.path.join(dir_path, "color"))


seqs = ['airplane', 'bag', 'bicycle', 'car', 'dancingshoe', 'diving', 'goldfish']

for dir_path in os.listdir("/datagrid/personal/rozumrus/vot2022/lt/sequences"):
	if not dir_path in seqs:
		continue

	# if dir_path[0] == ".":
	# 	continue
	# if dir_path == "grid_gen.py" or dir_path == "old_list.txt" or dir_path == "list.txt" or dir_path == "color_thick_to_init.py":
	#     continue

	# if os.path.exists(os.path.join(dir_path, "2p_grid_color")):
	# 	continue

	shutil.move(os.path.join(dir_path, "color"), os.path.join(dir_path, "initial_color"))
	# shutil.move(os.path.join(dir_path, "color_init1"), os.path.join(dir_path, "color"))

