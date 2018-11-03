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
from PIL import Image
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

depth = skimage.io.imread("/media/taylor/YH/Revisiting_Single_Depth_Estimation-master/data/nyu2_train/basement_0001b_out/2.png")
print(np.max(depth))
print(np.min(depth))

# matplotlib.image.imsave("/home/taylor/Mirror-Segmentation/data_640/test/mask/3_512x640_json/depth2.png", depth)

#
# a = np.array([[1, 255], [0, 1]], dtype=np.uint8)
# b = np.array([[1, 0], [87, 23]], dtype=np.uint8)
# c = np.logical_not(a)
# print(c)

#
# import numpy as np
#
# a = 3*np.ones([2 ,2])
# b = a - 1
# print(b)