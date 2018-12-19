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
import mhy.psp as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "psp_MSD")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_psp_MSD_all_60.h5")
# MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror20181111T1153/mirror_0045.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "MSD", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "MSD", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, "MSD_results", "MSD_PSP")
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
model = modellib.PSP(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

ACC_mirror = []
IOU = []
F = []
MAE = []
BER = []
NUM = []

start = time.time()
for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    visualize.save_mask_and_masked_image(imgname, image, r['mask'], OUTPUT_PATH)

    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################
    height = image.shape[0]
    width = image.shape[1]
    if height > width:
        predict_mask = r['mask'][0, :, :, 0][:, 64:576]
    elif height < width:
        predict_mask = r['mask'][0, :, :, 0][64:576, :]
    gt_mask = evaluation.get_mask_directly(imgname, MASK_DIR)
    print(np.shape(predict_mask))
    print(np.shape(gt_mask))

    acc_mirror = evaluation.accuracy_mirror(predict_mask, gt_mask)
    iou = evaluation.iou(predict_mask, gt_mask)
    f = evaluation.f_score(predict_mask, gt_mask)
    mae = evaluation.mae(predict_mask, gt_mask)
    ber = evaluation.ber(predict_mask, gt_mask)

    print("acc_mirror : {}".format(acc_mirror))
    print("iou : {}".format(iou))
    print("f : {}".format(f))
    print("mae : {}".format(mae))
    print("ber : {}".format(ber))

    ACC_mirror.append(acc_mirror)
    IOU.append(iou)
    F.append(f)
    MAE.append(mae)
    BER.append(ber)

    num = imgname.split("_")[0]
    NUM.append(int(num))

end = time.time()
print("Time is : {}".format(end - start))

mean_ACC_mirror = 100 * sum(ACC_mirror)/len(ACC_mirror)
mean_IOU = 100 * sum(IOU)/len(IOU)
mean_F = sum(F)/len(F)
mean_MAE = sum(MAE)/len(MAE)
mean_BER = 100 * sum(BER)/len(BER)

print(len(ACC_mirror))
print(len(IOU))
print(len(F))
print(len(MAE))
print(len(BER))

evaluation.data_write('./MSD_results/MSD_PSP.xlsx', [NUM, ACC_mirror, IOU, F, MAE, BER])

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}".
      format("mean_ACC_mirror", mean_ACC_mirror, "mean_IOU", mean_IOU, "mean_F", mean_F, "mean_MAE", mean_MAE, "mean_BER", mean_BER))

