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
IMAGE_DIR = os.path.join(ROOT_DIR, "MSD9", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "MSD9", "test", "mask")
# IMAGE_DIR = os.path.join("/home/iccd/Desktop/data", "image")
# MASK_DIR = os.path.join("/home/iccd/Desktop/data", "mask")
SHADOW_DIR = os.path.join(ROOT_DIR, "MSD9_results", "MSD9_PSP")
# SHADOW_DIR = "/home/iccd/BDRAR/ckpt/BDRAR/(BDRAR) sbu_prediction_3001"
TYPE = 0

if TYPE != 0:
    type_path = os.path.join("/home/iccd/data/types", str(TYPE))
    typelist = os.listdir(type_path)
    testlist = os.listdir(IMAGE_DIR)
    imglist = list(set(typelist) & set(testlist))
else:
    imglist = os.listdir(IMAGE_DIR)

print("Total {} test images".format(len(imglist)))

ACC = []
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

    acc = evaluation.accuracy_mirror(predict_mask, gt_mask)
    iou = evaluation.iou(predict_mask, gt_mask)
    f = evaluation.f_score(predict_mask, gt_mask)
    mae = evaluation.mae(predict_mask, gt_mask)
    ber = evaluation.ber(predict_mask, gt_mask)

    print("acc : {}".format(acc))
    print("iou : {}".format(iou))
    print("f : {}".format(f))
    print("mae : {}".format(mae))
    print("ber : {}".format(ber))

    ACC.append(acc)
    IOU.append(iou)
    F.append(f)
    MAE.append(mae)
    BER.append(ber)

    num = imgname.split("_")[0]
    NUM.append(int(num))

end = time.time()
print("Time is : {}".format(end - start))

mean_ACC = 100 * sum(ACC)/len(ACC)
mean_IOU = 100 * sum(IOU)/len(IOU)
mean_F = sum(F)/len(F)
mean_MAE = sum(MAE)/len(MAE)
mean_BER = 100 * sum(BER)/len(BER)

print(len(ACC))
print(len(IOU))
print(len(F))
print(len(MAE))
print(len(BER))

evaluation.data_write('BDRAR.xlsx', [NUM, ACC, IOU, F, MAE, BER])

print("{}, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.3f} \n{:20} {:.3f} \n{:20} {:.2f}\n".
      format(SHADOW_DIR, "mean_ACC", mean_ACC, "mean_IOU", mean_IOU, "mean_F", mean_F, "mean_MAE", mean_MAE, "mean_BER", mean_BER))

print("{:.2f} {:.2f} {:.3f} {:.3f} {:.2f}".format(mean_ACC, mean_IOU, mean_F, mean_MAE, mean_BER))


