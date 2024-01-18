from PIL import Image, ImageDraw
import os
import sys


def get_num_of_black_pixels(width, height, lineThickness, x, lineColor=(0, 0, 0)):
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

    rgb_image.save("white_w3.jpg", "JPEG")

    for x in range(width):

        for y in range(height):
            r, g, b = rgb_image.getpixel((x, y))
            if r == g == b == 0:
                black_pixel_count += 1
            
        
    return black_pixel_count / total



len_side, line_thick = sys.argv[1], sys.argv[2]
print("Percentage of unvisible pixels: ", get_num_of_black_pixels(432, 576, int(line_thick), int(len_side))*100)


#python3 get_unvis_pix.py 80 1