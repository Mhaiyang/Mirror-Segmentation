"""
  @Time    : 2018-10-22 18:08
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : Mirror-Segmentation
  @File    : test.py
  @Function: 
  
"""
# import tensorflow as tf
#
# a = 20 * tf.ones([1, 10, 10, 1])
# b = tf.count_nonzero(a, [1, 2])
# c = 327680 * tf.ones(tf.shape(b), dtype=tf.float32) - tf.cast(b, tf.float32)
# d = c / tf.cast(b, tf.float32)
# x = tf.squeeze(b)
#
# with tf.Session() as sess:
#     print(sess.run(tf.shape(b)))
#     print(sess.run(a))

import skimage.io
import numpy as np
import matplotlib.image
# from PIL import Image
# import matplotlib.pyplot as plt
# plt.set_cmap("jet")
#
# depth = Image.open("/home/taylor/Revisiting_Single_Depth_Estimation/data/mirror_depth2/6_512x640.npy")
# print(np.max(depth))

# depth = np.load("/home/taylor/Revisiting_Single_Depth_Estimation/data/mirror_depth2/6_512x640.npy")
#
# a = np.max(depth)
# b = np.min(depth)
# s = np.shape(depth)
# print(a)
# print(b)
# print(s)

# mask = skimage.io.imread("/home/taylor/Mirror-Segmentation/data_640/train/mask/1_512x640_json/depth.png")
# height = np.shape(mask)[0]
# width = np.shape(mask)[1]
# num_obj = np.max(mask)
# output = np.zeros([height, width], dtype=np.uint8)
# for index in range(num_obj):
#     """j is row and i is column"""
#     for i in range(height):
#         for j in range(width):
#             if mask[i, j]:
#                 output[i, j] = 255  # [height width channel] i.e. [h, w, c]
#
# skimage.io.imsave("/home/iccd/Desktop/output.png", output)

depth = skimage.io.imread("/media/taylor/mhy/depth_original/test/1_512x640.png")
print(np.max(depth))
print(np.min(depth))
print(depth.dtype)
depth = (depth.astype(np.float32)) / 255.0
print(np.max(depth))
print(np.min(depth))
print(depth.dtype)
# matplotlib.image.imsave("/home/taylor/Desktop/1.png", depth)

