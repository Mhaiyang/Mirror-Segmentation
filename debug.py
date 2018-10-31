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
# a = 160*tf.ones(2)
#
# with tf.Session() as sess:
#     print(sess.run(a))

# import skimage.io
import numpy as np
#
# edge = skimage.io.imread("/home/taylor/Mirror-Segmentation/data_640/train/mask/179_640x512_json/edge.png")
# print(np.max((edge/255).astype(np.uint8)))

a = np.array([[1, 255], [0, 1]], dtype=np.uint8)
b = np.array([[1, 0], [87, 23]], dtype=np.uint8)
c = np.logical_and(a, b)
print(c)