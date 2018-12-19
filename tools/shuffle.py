import os
import numpy as np
import random
import skimage.io

input_path = "/home/iccd/Desktop/data"
output_path = "/home/iccd/Desktop/MSD"

threshold = 2400
imglist = os.listdir(input_path + "/image")
print(len(imglist))
random.shuffle(imglist)

for i, imgname in enumerate(imglist):
    num, resolution = imgname.split("_")
    image = skimage.io.imread(input_path + "/image/" + imgname)
    mask = skimage.io.imread(input_path + "/mask/" + imgname[:-4] + ".png")
    if i < 2400:
        skimage.io.imsave(output_path + "/train/image/" + str(i + 1) + "_" + resolution, image)
        skimage.io.imsave(output_path + "/train/mask/" + str(i + 1) + "_" + resolution[:-4] + ".png", mask)
    else:
        skimage.io.imsave(output_path + "/test/image/" + str(i + 1) + "_" + resolution, image)
        skimage.io.imsave(output_path + "/test/mask/" + str(i + 1) + "_" + resolution[:-4] + ".png", mask)

print("ok")

