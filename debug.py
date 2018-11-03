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

depth = skimage.io.imread("/home/taylor/Revisiting_Single_Depth_Estimation/data/mirror_depth/7_640x512.jpg")
a = np.max(depth)
b = np.min(depth)
s = np.shape(depth)
print(a)
print(b)
print(s)
#
# a = np.array([[1, 255], [0, 1]], dtype=np.uint8)
# b = np.array([[1, 0], [87, 23]], dtype=np.uint8)
# c = np.logical_not(a)
# print(c)