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
IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test3", "image")
MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test3", "mask")
SHADOW_DIR = os.path.join(ROOT_DIR, "PiCANet")

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

IOU = []
ACC_all = []
ACC_mirror = []
BER = []

start = time.time()
for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################

    gt_mask = evaluation.get_mask(imgname, MASK_DIR)
    predict_mask = evaluation.get_predict_mask(imgname, SHADOW_DIR)

    iou = evaluation.iou(predict_mask, gt_mask)
    acc_all = evaluation.accuracy_all(predict_mask, gt_mask)
    acc_mirror = evaluation.accuracy_mirror(predict_mask, gt_mask)
    ber = evaluation.ber(predict_mask, gt_mask)
    # mse = skimage.measure.compare_mse(gt_depth, predict_depth)

    print("iou : {}".format(iou))
    print("acc_all : {}".format(acc_all))
    print("acc_mirror : {}".format(acc_mirror))
    print("ber : {}".format(ber))
    IOU.append(iou)
    ACC_all.append(acc_all)
    ACC_mirror.append(acc_mirror)
    BER.append(ber)

end = time.time()
print("Time is : {}".format(end - start))

mean_IOU = 100 * sum(IOU)/len(IOU)
mean_ACC_all = 100 * sum(ACC_all)/len(ACC_all)
mean_ACC_mirror = 100 * sum(ACC_mirror)/len(ACC_mirror)
mean_BER = 100 * sum(BER)/len(BER)
# mean_MSE = sum(MSE)/len(MSE)

print(len(IOU))
print(len(ACC_all))
print(len(ACC_mirror))
print(len(BER))

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f}".
      format("mean_IOU", mean_IOU, "mean_ACC_all", mean_ACC_all, "mean_ACC_mirror", mean_ACC_mirror, "mean_BER", mean_BER))



