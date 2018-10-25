"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import skimage.io
import mhy.visualize as visualize
import evaluation
from mirror import MirrorConfig
# Important, need change when test different models.
import mhy.fcn8 as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "fcn8")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_fcn8_heads.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data_640", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "data_640", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data_640', 'test', "output_fcn8")
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

pas = []
ious = []

for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    # Save results
    visualize.save_mask_and_masked_image(imgname, image, r['mask'], OUTPUT_PATH)

    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################

    gt_mask = evaluation.get_mask(imgname, MASK_DIR)
    gt_mask = gt_mask[:, :, 0]
    predict_mask_square = r['mask'][0, :, :, 0]

    height = gt_mask.shape[0]
    width = gt_mask.shape[1]
    if height > width:
        predict_mask = predict_mask_square[:, 64:576]
    elif height < width:
        predict_mask = predict_mask_square[64:576, :]

    print(predict_mask.shape)
    print(gt_mask.shape)
    pa = evaluation.pixel_accuracy(predict_mask, gt_mask)
    IoU = evaluation.IoU(predict_mask, gt_mask)

    print("pixel accuracy : {}".format(pa))
    print("IOU            : {}".format(IoU))
    pas.append(pa)
    ious.append(IoU)

pixel_accuracy = 100 * sum(pas)/len(pas)
mean_iou = 100 * sum(ious)/len(ious)

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f}".format("pixel_accuracy", pixel_accuracy,
                                                                "mean_iou", mean_iou))





