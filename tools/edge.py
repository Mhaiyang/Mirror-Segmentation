"""
  @Time    : 2018-10-28 00:09
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : Mirror-Segmentation
  @File    : edge.py
  @Function: 
  
"""

import os
import numpy as np
import cv2
from PIL import Image

# DATA_DIR = "/home/iccd/Mirror-Segmentation/data_640/val"
DATA_DIR = "/root/data_640/test"
IMAGE_DIR = os.path.join(DATA_DIR, "image")

imglist = os.listdir(IMAGE_DIR)
print("Total {} masks will be extracted edge!".format(len(imglist)))

q = 0
for imgname in imglist:
    q += 1
    mask_path = DATA_DIR + "/mask/" + imgname[:-4] + "_json/label8.png"
    edge_path = DATA_DIR + "/mask/" + imgname[:-4] + "_json/edge.png"

    mask = Image.open(mask_path)
    num_obj = np.max(mask)

    width, height = mask.size
    gt_mask = np.zeros([height, width, 1], dtype=np.uint8)
    for index in range(num_obj):
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i, j))
                if at_pixel == index + 1:
                    gt_mask[j, i, 0] = 255

    edge = cv2.Canny(gt_mask, 0, 255)
    edge = np.where(edge != 0, 255, 0).astype(np.uint8)

    cv2.imwrite(edge_path, edge)
    print("{}  {}".format(q, edge_path))
