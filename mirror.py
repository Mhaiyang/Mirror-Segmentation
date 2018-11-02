import os
import numpy as np
from PIL import Image
from mhy.config import Config
import mhy.utils as utils
import skimage.io
import skimage.color


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
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    BACKBONE = "resnet101"
    # Pretrained_Model_Path = "/home/taylor/Mirror-Segmentation/pspnet101_voc2012.h5"
    Pretrained_Model_Path = "/root/pspnet101_voc2012.h5"

    BACKBONE_STRIDES = [4, 8, 16, 32, 64]   # for compute pyramid feature size

    LOSS_WEIGHTS = {
        "mask_loss": 1.,
        # "edge_loss": 10.,
        "depth_loss": 0.0001,
    }

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = int(3465/(GPU_COUNT*IMAGES_PER_GPU))

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = int(247/(GPU_COUNT*IMAGES_PER_GPU))

    # Learning rate
    LEARNING_RATE = 0.01


# Dataset
class MirrorDataset(utils.Dataset):

    def get_obj_index(self, image):
        """Get the number of instance in the image
        """
        n = np.max(image)
        return n

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
        for i in range(count):
            filestr = imglist[i].split(".")[0]  # 10.jpg for example
            image_path = img_folder + "/" + imglist[i]
            mask_path = mask_folder + "/" + filestr + "_json/label8.png"
            edge_path = mask_folder + "/" + filestr + "_json/edge.png"
            depth_path = mask_folder + "/" + filestr + "_json/depth.png"
            if not os.path.exists(mask_path):
                print("{} is incorrect".format(filestr))
                continue
            img = Image.open(mask_path)
            width, height = img.size
            self.add_image(image_id=i, image_path=image_path, width=width, height=height,
                           mask_path=mask_path, edge_path=edge_path, depth_path=depth_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['image_path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        global iter_num
        info = self.image_info[image_id]
        image = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(image)
        mask = np.zeros([info['height'], info['width']], dtype=np.uint8)
        for index in range(num_obj):
            """j is row and i is column"""
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i] = 1  # [height width channel] i.e. [h, w]

        return mask

    def load_edge(self, image_id):
        info = self.image_info[image_id]
        edge_gray = skimage.io.imread(info["edge_path"])
        edge = (edge_gray[:, :]/255).astype(np.uint8)

        return edge

    def load_depth(self, image_id):
        """Load the specified depth and return a [H, W] Numpy array"""

        depth = skimage.io.imread(self.image_info[image_id]['depth_path'])

        return depth.astype(np.uint8)




