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
from mirror import MirrorConfig
# Important, need change when test different models.
import mhy.psp_edge_v3_no as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "psp_edge_v3_no")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_psp_edge_v3_no_all_40.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data_640', 'test', "output_edge_v3_no_40")
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
model = modellib.PSP_EDGE(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

IOU = []
ACC = []
BER = []
# PSNR = []
# SSIM = []

for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    # Save mask and masked image.
    visualize.save_mask_and_masked_image(imgname, image, r['mask'], OUTPUT_PATH)

    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################

    gt_mask = evaluation.get_mask(imgname, MASK_DIR)
    gt_depth = skimage.io.imread(MASK_DIR + "/" + imgname[:-4] + "_json/depth.png")
    predict_mask_square = r['mask'][0, :, :, 0]

    height = gt_mask.shape[0]
    width = gt_mask.shape[1]
    if height > width:
        predict_mask = predict_mask_square[:, 64:576]
    elif height < width:
        predict_mask = predict_mask_square[64:576, :]

    # # # if have edge branch
    # if height > width:
    #     # predict_semantic = r["semantic"][0, :, :, 0][:, 64:576]
    #     # predict_edge = r["edge"][0, :, :, 0][:, 64:576]
    #     predict_depth = r["depth"][0, :, :, 0][:, 64:576]
    # elif height < width:
    #     # predict_semantic = r["semantic"][0, :, :, 0][64:576, :]
    #     # predict_edge = r["edge"][0, :, :, 0][64:576, :]
    #     predict_depth = r["depth"][0, :, :, 0][64:576, :]
    # # skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname[:-4]+"_edge.jpg"), (255 * predict_edge).astype(np.uint8))
    # skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname[:-4]+"_depth.jpg"), predict_depth.astype(np.uint8))

    iou = evaluation.iou(predict_mask, gt_mask)
    acc = evaluation.accuracy(predict_mask, gt_mask)
    ber = evaluation.ber(predict_mask, gt_mask)
    # psnr = skimage.measure.compare_psnr(gt_depth, predict_depth)
    # ssim = skimage.measure.compare_ssim(gt_depth, predict_depth)

    print("iou : {}".format(iou))
    print("acc : {}".format(acc))
    print("ber : {}".format(ber))
    # print("psnr : {}".format(psnr))
    # print("ssim : {}".format(ssim))
    IOU.append(iou)
    ACC.append(acc)
    BER.append(ber)
    # PSNR.append(psnr)
    # SSIM.append(ssim)

mean_IOU = 100 * sum(IOU)/len(IOU)
mean_ACC = 100 * sum(ACC)/len(ACC)
mean_BER = 100 * sum(BER)/len(BER)
# mean_PSNR = sum(PSNR)/len(PSNR)
# mean_SSIM = sum(SSIM)/len(SSIM)

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f}".
      format("mean_IOU", mean_IOU, "mean_ACC", mean_ACC, "mean_BER", mean_BER))
# print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.4f}".
#       format("mean_IOU", mean_IOU, "mean_ACC", mean_ACC, "mean_BER", mean_BER,
#              "mean_PSNR", mean_PSNR, "mean_SSIM", mean_SSIM))





