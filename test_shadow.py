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
import evaluation


# Directories of the project
ROOT_DIR = os.getcwd()
# IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test3", "image")
# MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test3", "mask")
IMAGE_DIR = os.path.join(ROOT_DIR, "MSD", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "MSD", "test", "mask")
# IMAGE_DIR = os.path.join("/home/iccd/Desktop/data", "image")
# MASK_DIR = os.path.join("/home/iccd/Desktop/data", "mask")
SHADOW_DIR = os.path.join(ROOT_DIR, "MSD_RA")

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
    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################

    # gt_mask = evaluation.get_mask(imgname, MASK_DIR)
    gt_mask = evaluation.get_mask_directly(imgname, MASK_DIR)
    predict_mask = evaluation.get_predict_mask(imgname, SHADOW_DIR)

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

evaluation.data_write('MSD_RAS.xlsx', [NUM, ACC_mirror, IOU, F, MAE, BER])

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}".
      format("mean_ACC_mirror", mean_ACC_mirror, "mean_IOU", mean_IOU, "mean_F", mean_F, "mean_MAE", mean_MAE, "mean_BER", mean_BER))



