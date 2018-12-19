import os
import numpy as np
import skimage.io
import shutil

input_path = "/home/iccd/Desktop/MSD_json/test_disorder"
output_path = "/home/iccd/Desktop/MSD_json/test"

imglist = os.listdir(input_path + "/image")
for i, imgname in enumerate(imglist):
    print(imgname)
    num, resolution = imgname.split("_")
    image = skimage.io.imread(input_path + "/image/" + imgname)
    # mask = skimage.io.imread(input_path + "/mask/" + imgname[:-4] + ".png")

    skimage.io.imsave(output_path + "/image/" + str(i + 1) + "_" + resolution, image)
    # skimage.io.imsave(output_path + "/mask/" + str(i + 1) + "_" + resolution[:-4] + ".png", mask)
    shutil.copytree(input_path + "/mask/" + imgname[:-4] + "_json", output_path + "/mask/" + str(i + 1) + "_" + resolution[:-4] + "_json")