import os
import numpy as np
import skimage.io
import skimage.transform

input_path = "/home/iccd/Desktop/test/picture"
output_path = "/home/iccd/Desktop/test/picture_output"
if not os.path.exists(output_path):
    os.mkdir(output_path)

imglist = os.listdir(input_path)
for i, imgname in enumerate(imglist):
    print(i)
    old_image = skimage.io.imread(os.path.join(input_path, imgname))
    height = np.shape(old_image)[0]
    width = np.shape(old_image)[1]
    if width > height:
        fixed_size = (640, 512)
    else:
        fixed_size = (512, 640)
    new_image = skimage.transform.resize(old_image, (fixed_size[1], fixed_size[0]), order=3)
    skimage.io.imsave(os.path.join(output_path, imgname), new_image)