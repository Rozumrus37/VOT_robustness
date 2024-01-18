from PIL import Image, ImageDraw
import os

gridDivisions = 16


lineColor = 'black'

lineThickness = 5


for filename_in in os.listdir("sequences"):
	if filename_in[0] != "." and filename_in[len(filename_in)-3:] != "txt":
		for filename in os.listdir("sequences/" + filename_in):
		    imagepath = "sequences/" + filename_in + "/color/00000001.jpg"

		    im = Image.open(imagepath)
		    filename = str(os.path.splitext(os.path.basename(imagepath))[0])

		    width = im.size[0]
		    height = im.size[1]

		    print(width, height)
