"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import numpy as np
import skimage.io
import skimage.measure
import mhy.visualize as visualize
import evaluation
import time
from mirror import MirrorConfig
# Important, need change when test different models.
import mhy.fcn8 as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "fcn8")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_fcn8_heads_80.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test2", "image")
MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test2", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data_640', 'test2', "fcn8")
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
model = modellib.FCN8(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

IOU = []
ACC = []
BER = []
MSE = []

start = time.time()
for i, imgname in enumerate(imglist):

    print(i)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    results = model.detect(imgname, [image], verbose=1)

end = time.time()

print("Time is : {}".format(end - start))


