"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import numpy as np
import skimage.io
import skimage.measure
import time
import mhy.visualize as visualize
import evaluation
from mirror import MirrorConfig
# Important, need change when test different models.
import mhy.psp_edge_depth_v14 as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "psp_edge_depth_v14")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_psp_edge_depth_v14_all_45.h5")
# MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror20181111T1153/mirror_0045.h5")
# IMAGE_DIR = os.path.join("/home/iccd/Desktop/test/4")
IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test3", "image")
MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test3", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data_640', 'test3', "psp_edge_depth_v14_0045_prob")
# OUTPUT_PATH = os.path.join("/home/iccd/Desktop/test/output_4")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


# Configurations
class InferenceConfig(MirrorConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # Up to now, batch size must be one.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.PSP_EDGE_DEPTH(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))


start = time.time()
for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    predict_mask_square = r['mask'][0, :, :, 0]
    height = image.shape[0]
    width = image.shape[1]
    if height > width:
        predict_mask = predict_mask_square[:, 64:576]
    elif height < width:
        predict_mask = predict_mask_square[64:576, :]
    skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname), predict_mask)






