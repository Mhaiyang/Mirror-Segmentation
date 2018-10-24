"""
  @Time    : 2018-7-28 04:56
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : resize_to_640.py
  @Function: resize image to 640

  @Last modified: 2018-10-24
  
"""
import os
from PIL import Image
from skimage import io, transform

# Important, need modify.
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../data", "test"))
IMAGE_DIR = os.path.join(DATA_DIR, "image")
MASK_DIR = os.path.join(DATA_DIR, "mask")

# Important, need modify.
OUTPUT_DIR = os.path.join(DATA_DIR, "../../data_640", "test")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    os.mkdir(os.path.join(OUTPUT_DIR, "image"))
    os.mkdir(os.path.join(OUTPUT_DIR, "mask"))

imglist = os.listdir(IMAGE_DIR)
print("Total {} images will be resized to 640!".format(len(imglist)))

for i, imgname in enumerate(imglist):
    print(i, imgname)
    filestr = imgname.split(".")[0]
    image_path = IMAGE_DIR + "/" + imgname
    mask_path = MASK_DIR + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")

    mask = Image.open(mask_path)
    image = io.imread(image_path)

    width, height = mask.size
    if width > height:
        fixed_size = (640, 512)
    else:
        fixed_size = (512, 640)
    fixed_image = transform.resize(image, (fixed_size[1], fixed_size[0]), order=3)
    io.imsave(OUTPUT_DIR + "/image/" + filestr + "_" + str(fixed_size[0]) + "x" + str(fixed_size[1]) + ".jpg", fixed_image)
    fixed_mask = mask.resize(fixed_size, Image.BICUBIC)
    if not os.path.exists(OUTPUT_DIR + "/mask/" + filestr + "_" + str(fixed_size[0]) + "x" + str(fixed_size[1]) + "_json"):
        os.mkdir(OUTPUT_DIR + "/mask/" + filestr + "_" + str(fixed_size[0]) + "x" + str(fixed_size[1]) + "_json")
    fixed_mask.save(OUTPUT_DIR + "/mask/" + filestr + "_" + str(fixed_size[0]) + "x" + str(fixed_size[1]) + "_json/label8.png")


