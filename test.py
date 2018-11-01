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
import mhy.psp_edge_depth as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "psp_edge_depth")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_psp_edge_depth_all_40.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data_640', 'test', "output_edge_depth_40")
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

pas = []
ious = []
PSNR = []
SSIM = []


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

    # # if have edge branch
    if height > width:
        # predict_semantic = r["semantic"][0, :, :, 0][:, 64:576]
        predict_edge = r["edge"][0, :, :, 0][:, 64:576]
        predict_depth = r["depth"][0, :, :, 0][:, 64:576]
    elif height < width:
        # predict_semantic = r["semantic"][0, :, :, 0][64:576, :]
        predict_edge = r["edge"][0, :, :, 0][64:576, :]
        predict_depth = r["depth"][0, :, :, 0][64:576, :]
    skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname[:-4]+"_edge.jpg"), (255 * predict_edge).astype(np.uint8))
    skimage.io.imsave(os.path.join(OUTPUT_PATH, imgname[:-4]+"_depth.jpg"), predict_depth.astype(np.uint8))

    pa = evaluation.pixel_accuracy(predict_mask, gt_mask)
    IoU = evaluation.IoU(predict_mask, gt_mask)
    psnr = skimage.measure.compare_psnr(gt_depth, predict_depth)
    ssim = skimage.measure.compare_ssim(gt_depth, predict_depth)

    print("pixel accuracy : {}".format(pa))
    print("IOU            : {}".format(IoU))
    print("psnr           : {}".format(psnr))
    print("ssim           : {}".format(ssim))
    pas.append(pa)
    ious.append(IoU)
    PSNR.append(psnr)
    SSIM.append(ssim)

pixel_accuracy = 100 * sum(pas)/len(pas)
mean_iou = 100 * sum(ious)/len(ious)
mean_psnr = sum(PSNR)/len(PSNR)
mean_ssim = sum(SSIM)/len(SSIM)

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.4f}".
      format("pixel_accuracy", pixel_accuracy, "mean_iou", mean_iou, "mean_psnr", mean_psnr, "mean_ssim", mean_ssim))





