import os
import numpy as np
from PIL import Image
from mhy.config import Config
import mhy.utils as utils
import yaml
import skimage.io


# Configurations
class MirrorConfig(Config):
    """Configuration for training on the mirror dataset.
    Derives from the base Config class and overrides values specific
    to the mirror dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Mirror"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 mirror

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    BACKBONE = "resnet101"
    Pretrained_Model_Path = "/home/taylor/Mirror-Segmentation/resnet50.h5"

    BACKBONE_STRIDES = [4, 8, 16, 32, 64]   # for compute pyramid feature size

    LOSS_WEIGHTS = {
        "mask_loss": 1.,
        # "rpn_bbox_loss": 1.,
    }

    # For One, Two, Three, and their combination. National Day.
    # EDGE_SHAPE : [h, w]
    CLASSIFY_POOL_SIZE = 7
    MASK_POOL_SIZE = [32, 16, 8, 4]
    MASK_SHAPE = [32, 32]
    # EDGE_SHAPE = [32, 32]

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = int(3465/(GPU_COUNT*IMAGES_PER_GPU))

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = int(741/(GPU_COUNT*IMAGES_PER_GPU))

    # skip detection with <x% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    # Learning rate
    LEARNING_RATE = 0.001


# Dataset
class MirrorDataset(utils.Dataset):

    def get_obj_index(self, image):
        """Get the number of instance in the image
        """
        n = np.max(image)
        return n

    def from_yaml_get_class(self,image_id):
        """Translate the yaml file to get label """
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            """j is row and i is column"""
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1   # [height width channel] i.e. [h, w, c]
        return mask

    def load_mirror(self, count, img_folder, mask_folder, imglist):
        self.add_class("Mirror", 1, "mirror")
        # self.add_class("Mirror", 2, "reflection")
        for i in range(count):
            filestr = imglist[i].split(".")[0]  # 10.jpg for example
            mask_path = mask_folder + "/" + filestr + "_json/label8.png"
            edge_path = mask_folder + "/" + filestr + "_json/edge.png"
            yaml_path = mask_folder + "/" + filestr + "_json/info.yaml"
            if not os.path.exists(mask_path):
                print("{} is incorrect".format(filestr))
                continue
            img = Image.open(mask_path)
            width, height = img.size
            self.add_image("Mirror", image_id=i, path=img_folder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path, edge_path=edge_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        global iter_num
        info = self.image_info[image_id]
        image = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(image)
        mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        for index in range(num_obj):
            """j is row and i is column"""
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, 0] = 1  # [height width channel] i.e. [h, w, 1]

        return mask

    def load_edge(self, image_id):
        info = self.image_info[image_id]
        edge = skimage.io.imread(info["edge_path"])

        return edge




