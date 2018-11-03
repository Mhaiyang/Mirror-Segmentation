"""
  @Time    : 2018-10-31 22:55
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : Mirror-Segmentation
  @File    : depth.py
  @Function: Generate mirror depth ground truth.
  
"""
import os
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

PREDICT_DEPTH_DIR = "/media/taylor/mhy/depth/val/"
OUTPUT_DIR = "/media/taylor/mhy/depth_mirror/val/"
IMAGE_DIR = "/home/taylor/Mirror-Segmentation/data_640/val/image/"
MASK_DIR = "/home/taylor/Mirror-Segmentation/data_640/val/mask/"
# PREDICT_DEPTH_DIR = "/root/FCRN/test_depth/"
# IMAGE_DIR = "/root/data_640/test/image/"
# MASK_DIR = "/root/data_640/test/mask/"

imglist = os.listdir(IMAGE_DIR)
print("{} predict depth map will be processed.".format(len(imglist)))

for n, imgname in enumerate(imglist):
    print("{} ########### {} #########".format(n, imgname))

    filestr = imgname.split(".")[0]     # 1_512x640.jpg
    predict_depth_path = PREDICT_DEPTH_DIR + filestr + ".npy"
    output_depth_path1 = OUTPUT_DIR + filestr + ".npy"
    output_depth_path2 = OUTPUT_DIR + filestr + ".png"
    mask_path = MASK_DIR + filestr + "_json/label8.png"

    predict_depth = np.load(predict_depth_path)
    mask = skimage.io.imread(mask_path)
    height = mask.shape[0]
    width = mask.shape[1]
    num_obj = np.max(mask)

    output_depth = predict_depth
    for index in range(num_obj):

        mirror_depth = []
        for j in range(height):
            for i in range(width):
                if mask[j, i] == index + 1:
                    mirror_depth.append(predict_depth[j, i])

        mean_mirror_depth = sum(mirror_depth) / len(mirror_depth)
        print("mean depth of mirror {} is : {}".format(index, mean_mirror_depth))

        for j in range(height):
            for i in range(width):
                if mask[j, i] == index + 1:
                    output_depth[j, i] = mean_mirror_depth

    np.save(output_depth_path1, output_depth)
    matplotlib.image.imsave(output_depth_path2, output_depth)




