import os
import numpy as np
import skimage.io

input_path = "/home/iccd/Mirror-Segmentation/data_640/test3"
output_path = "/home/iccd/Desktop/data_json"

initial_num = 3712
imglist = os.listdir(input_path + "/image")
print(initial_num, len(imglist))
for i, imgname in enumerate(imglist):
    num, resolution = imgname.split("_")
    num = int(num)
    image = skimage.io.imread(input_path + "/image/" + imgname)
    mask = skimage.io.imread(input_path + "/mask/" + imgname[:-4] + "_json/label8.png")
    edge = skimage.io.imread(input_path + "/mask/" + imgname[:-4] + "_json/edge.png")

    if not os.path.exists(output_path + "/mask/" + str(num + initial_num) + "_" + resolution[:-4] + "_json"):
        os.mkdir(output_path + "/mask/" + str(num + initial_num) + "_" + resolution[:-4] + "_json")
    skimage.io.imsave(output_path + "/image/" + str(num + initial_num) + "_" + resolution, image)
    skimage.io.imsave(output_path + "/mask/" + str(num + initial_num) + "_" + resolution[:-4] + "_json/label8.png", mask)
    skimage.io.imsave(output_path + "/mask/" + str(num + initial_num) + "_" + resolution[:-4] + "_json/edge.png", edge)

print("ok")

