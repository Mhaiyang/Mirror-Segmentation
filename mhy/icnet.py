"""
  @Time    : 2018-9-1 05:14
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : mirror
  @File    : segnet.py
  @Function: Segnet

"""

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM


from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.models import Model


from mhy import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when inferencing
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layres
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layres
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    C1 = x = KL.Activation('relu')(x)
    # Stage 2
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  Loss Functions
############################################################
def mask_loss_graph_4(input_gt_mask, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, height, width, 1]. bool. Convert it to
        a float32 tensor of values 0 or 1.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    gt_mask = tf.expand_dims(input_gt_mask, -1)
    gt_mask = tf.image.resize_bilinear(tf.cast(gt_mask, tf.float32), 160 * tf.ones(2, dtype=tf.int32))
    gt_mask = tf.round(gt_mask)
    gt_mask = tf.squeeze(gt_mask, -1)

    pred_masks = K.squeeze(pred_masks, -1)

    loss = K.binary_crossentropy(target=gt_mask, output=pred_masks)
    loss = K.mean(loss)

    return loss


def mask_loss_graph_8(input_gt_mask, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, height, width, 1]. bool. Convert it to
        a float32 tensor of values 0 or 1.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    gt_mask = tf.expand_dims(input_gt_mask, -1)
    gt_mask = tf.image.resize_bilinear(tf.cast(gt_mask, tf.float32), 80 * tf.ones(2, dtype=tf.int32))
    gt_mask = tf.round(gt_mask)
    gt_mask = tf.squeeze(gt_mask, -1)

    pred_masks = K.squeeze(pred_masks, -1)

    loss = K.binary_crossentropy(target=gt_mask, output=pred_masks)
    loss = K.mean(loss)

    return loss


def mask_loss_graph_16(input_gt_mask, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, height, width, 1]. bool. Convert it to
        a float32 tensor of values 0 or 1.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    gt_mask = tf.expand_dims(input_gt_mask, -1)
    gt_mask = tf.image.resize_bilinear(tf.cast(gt_mask, tf.float32), 40 * tf.ones(2, dtype=tf.int32))
    gt_mask = tf.round(gt_mask)
    gt_mask = tf.squeeze(gt_mask, -1)

    pred_masks = K.squeeze(pred_masks, -1)

    loss = K.binary_crossentropy(target=gt_mask, output=pred_masks)
    loss = K.mean(loss)

    return loss

############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask = dataset.load_mask(image_id)
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)
    mask = np.round(mask)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        k = random.randint(0, 1, 2, 3)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

    return image, mask


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=augmentation,
                              use_mini_mask=config.USE_MINI_MASK)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1]), dtype=gt_masks.dtype)

            # Add to batch
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_masks[b] = gt_masks

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_masks]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

############################################################
#  Network Class
############################################################

class ICNET(object):
    """
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build network architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference'], "Mode must be 'training' or 'inference'!"

        # Image size must be dividable by 2 multiple times
        # 640 x 640
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 320, 480, 512, 640, etc. ")

        # Inputs
        input_image = KL.Input(
            shape=[640, 640, 3], name="input_image", dtype=tf.float32)

        if mode == "training":
            # 1. GT Masks [batch, height, width]
            input_gt_mask = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]], name="input_gt_mask", dtype=tf.uint8)

        # Build ICNet.
        # (1/2)
        x = input_image
        y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)),
                   name='data_sub2')(x)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_1_3x3_s2')(y)
        y = BatchNormalization(name='conv1_1_3x3_s2_bn')(y)
        y = Conv2D(32, 3, padding='same', activation='relu', name='conv1_2_3x3')(y)
        y = BatchNormalization(name='conv1_2_3x3_s2_bn')(y)
        y = Conv2D(64, 3, padding='same', activation='relu', name='conv1_3_3x3')(y)
        y = BatchNormalization(name='conv1_3_3x3_bn')(y)
        y_ = MaxPooling2D(pool_size=3, strides=2, name='pool1_3x3_s2')(y)

        y = Conv2D(128, 1, name='conv2_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv2_1_1x1_proj_bn')(y)
        y_ = Conv2D(32, 1, activation='relu', name='conv2_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv2_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name='padding1')(y_)
        y_ = Conv2D(32, 3, activation='relu', name='conv2_1_3x3')(y_)
        y_ = BatchNormalization(name='conv2_1_3x3_bn')(y_)
        y_ = Conv2D(128, 1, name='conv2_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv2_1_1x1_increase_bn')(y_)
        y = Add(name='conv2_1')([y, y_])
        y_ = Activation('relu', name='conv2_1/relu')(y)

        y = Conv2D(32, 1, activation='relu', name='conv2_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv2_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding2')(y)
        y = Conv2D(32, 3, activation='relu', name='conv2_2_3x3')(y)
        y = BatchNormalization(name='conv2_2_3x3_bn')(y)
        y = Conv2D(128, 1, name='conv2_2_1x1_increase')(y)
        y = BatchNormalization(name='conv2_2_1x1_increase_bn')(y)
        y = Add(name='conv2_2')([y, y_])
        y_ = Activation('relu', name='conv2_2/relu')(y)

        y = Conv2D(32, 1, activation='relu', name='conv2_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv2_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding3')(y)
        y = Conv2D(32, 3, activation='relu', name='conv2_3_3x3')(y)
        y = BatchNormalization(name='conv2_3_3x3_bn')(y)
        y = Conv2D(128, 1, name='conv2_3_1x1_increase')(y)
        y = BatchNormalization(name='conv2_3_1x1_increase_bn')(y)
        y = Add(name='conv2_3')([y, y_])
        y_ = Activation('relu', name='conv2_3/relu')(y)

        y = Conv2D(256, 1, strides=2, name='conv3_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv3_1_1x1_proj_bn')(y)
        y_ = Conv2D(64, 1, strides=2, activation='relu', name='conv3_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv3_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(name='padding4')(y_)
        y_ = Conv2D(64, 3, activation='relu', name='conv3_1_3x3')(y_)
        y_ = BatchNormalization(name='conv3_1_3x3_bn')(y_)
        y_ = Conv2D(256, 1, name='conv3_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv3_1_1x1_increase_bn')(y_)
        y = Add(name='conv3_1')([y, y_])
        z = Activation('relu', name='conv3_1/relu')(y)

        # (1/4)
        y_ = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) // 2, int(x.shape[2]) // 2)),
                    name='conv3_1_sub4')(z)
        y = Conv2D(64, 1, activation='relu', name='conv3_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding5')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_2_3x3')(y)
        y = BatchNormalization(name='conv3_2_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_2_1x1_increase')(y)
        y = BatchNormalization(name='conv3_2_1x1_increase_bn')(y)
        y = Add(name='conv3_2')([y, y_])
        y_ = Activation('relu', name='conv3_2/relu')(y)

        y = Conv2D(64, 1, activation='relu', name='conv3_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding6')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_3_3x3')(y)
        y = BatchNormalization(name='conv3_3_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_3_1x1_increase')(y)
        y = BatchNormalization(name='conv3_3_1x1_increase_bn')(y)
        y = Add(name='conv3_3')([y, y_])
        y_ = Activation('relu', name='conv3_3/relu')(y)

        y = Conv2D(64, 1, activation='relu', name='conv3_4_1x1_reduce')(y_)
        y = BatchNormalization(name='conv3_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(name='padding7')(y)
        y = Conv2D(64, 3, activation='relu', name='conv3_4_3x3')(y)
        y = BatchNormalization(name='conv3_4_3x3_bn')(y)
        y = Conv2D(256, 1, name='conv3_4_1x1_increase')(y)
        y = BatchNormalization(name='conv3_4_1x1_increase_bn')(y)
        y = Add(name='conv3_4')([y, y_])
        y_ = Activation('relu', name='conv3_4/relu')(y)

        y = Conv2D(512, 1, name='conv4_1_1x1_proj')(y_)
        y = BatchNormalization(name='conv4_1_1x1_proj_bn')(y)
        y_ = Conv2D(128, 1, activation='relu', name='conv4_1_1x1_reduce')(y_)
        y_ = BatchNormalization(name='conv4_1_1x1_reduce_bn')(y_)
        y_ = ZeroPadding2D(padding=2, name='padding8')(y_)
        y_ = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_1_3x3')(y_)
        y_ = BatchNormalization(name='conv4_1_3x3_bn')(y_)
        y_ = Conv2D(512, 1, name='conv4_1_1x1_increase')(y_)
        y_ = BatchNormalization(name='conv4_1_1x1_increase_bn')(y_)
        y = Add(name='conv4_1')([y, y_])
        y_ = Activation('relu', name='conv4_1/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding9')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_2_3x3')(y)
        y = BatchNormalization(name='conv4_2_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_2_1x1_increase')(y)
        y = BatchNormalization(name='conv4_2_1x1_increase_bn')(y)
        y = Add(name='conv4_2')([y, y_])
        y_ = Activation('relu', name='conv4_2/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding10')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_3_3x3')(y)
        y = BatchNormalization(name='conv4_3_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_3_1x1_increase')(y)
        y = BatchNormalization(name='conv4_3_1x1_increase_bn')(y)
        y = Add(name='conv4_3')([y, y_])
        y_ = Activation('relu', name='conv4_3/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_4_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_4_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding11')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_4_3x3')(y)
        y = BatchNormalization(name='conv4_4_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_4_1x1_increase')(y)
        y = BatchNormalization(name='conv4_4_1x1_increase_bn')(y)
        y = Add(name='conv4_4')([y, y_])
        y_ = Activation('relu', name='conv4_4/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_5_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_5_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding12')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_5_3x3')(y)
        y = BatchNormalization(name='conv4_5_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_5_1x1_increase')(y)
        y = BatchNormalization(name='conv4_5_1x1_increase_bn')(y)
        y = Add(name='conv4_5')([y, y_])
        y_ = Activation('relu', name='conv4_5/relu')(y)

        y = Conv2D(128, 1, activation='relu', name='conv4_6_1x1_reduce')(y_)
        y = BatchNormalization(name='conv4_6_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=2, name='padding13')(y)
        y = Conv2D(128, 3, dilation_rate=2, activation='relu', name='conv4_6_3x3')(y)
        y = BatchNormalization(name='conv4_6_3x3_bn')(y)
        y = Conv2D(512, 1, name='conv4_6_1x1_increase')(y)
        y = BatchNormalization(name='conv4_6_1x1_increase_bn')(y)
        y = Add(name='conv4_6')([y, y_])
        y = Activation('relu', name='conv4_6/relu')(y)

        y_ = Conv2D(1024, 1, name='conv5_1_1x1_proj')(y)
        y_ = BatchNormalization(name='conv5_1_1x1_proj_bn')(y_)
        y = Conv2D(256, 1, activation='relu', name='conv5_1_1x1_reduce')(y)
        y = BatchNormalization(name='conv5_1_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding14')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_1_3x3')(y)
        y = BatchNormalization(name='conv5_1_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_1_1x1_increase')(y)
        y = BatchNormalization(name='conv5_1_1x1_increase_bn')(y)
        y = Add(name='conv5_1')([y, y_])
        y_ = Activation('relu', name='conv5_1/relu')(y)

        y = Conv2D(256, 1, activation='relu', name='conv5_2_1x1_reduce')(y_)
        y = BatchNormalization(name='conv5_2_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding15')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_2_3x3')(y)
        y = BatchNormalization(name='conv5_2_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_2_1x1_increase')(y)
        y = BatchNormalization(name='conv5_2_1x1_increase_bn')(y)
        y = Add(name='conv5_2')([y, y_])
        y_ = Activation('relu', name='conv5_2/relu')(y)

        y = Conv2D(256, 1, activation='relu', name='conv5_3_1x1_reduce')(y_)
        y = BatchNormalization(name='conv5_3_1x1_reduce_bn')(y)
        y = ZeroPadding2D(padding=4, name='padding16')(y)
        y = Conv2D(256, 3, dilation_rate=4, activation='relu', name='conv5_3_3x3')(y)
        y = BatchNormalization(name='conv5_3_3x3_bn')(y)
        y = Conv2D(1024, 1, name='conv5_3_1x1_increase')(y)
        y = BatchNormalization(name='conv5_3_1x1_increase_bn')(y)
        y = Add(name='conv5_3')([y, y_])
        y = Activation('relu', name='conv5_3/relu')(y)

        h, w = y.shape[1:3].as_list()
        pool1 = AveragePooling2D(pool_size=(h, w), strides=(h, w), name='conv5_3_pool1')(y)
        pool1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool1_interp')(pool1)
        pool2 = AveragePooling2D(pool_size=(h / 2, w / 2), strides=(h // 2, w // 2), name='conv5_3_pool2')(y)
        pool2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool2_interp')(pool2)
        pool3 = AveragePooling2D(pool_size=(h / 3, w / 3), strides=(h // 3, w // 3), name='conv5_3_pool3')(y)
        pool3 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool3_interp')(pool3)
        pool6 = AveragePooling2D(pool_size=(h / 4, w / 4), strides=(h // 4, w // 4), name='conv5_3_pool6')(y)
        pool6 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(h, w)), name='conv5_3_pool6_interp')(pool6)

        y = Add(name='conv5_3_sum')([y, pool1, pool2, pool3, pool6])
        y = Conv2D(256, 1, activation='relu', name='conv5_4_k1')(y)
        y = BatchNormalization(name='conv5_4_k1_bn')(y)
        aux_1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                       name='conv5_4_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        y = Conv2D(128, 3, dilation_rate=2, name='conv_sub4')(y)
        y = BatchNormalization(name='conv_sub4_bn')(y)
        y_ = Conv2D(128, 1, name='conv3_1_sub2_proj')(z)
        y_ = BatchNormalization(name='conv3_1_sub2_proj_bn')(y_)
        y = Add(name='sub24_sum')([y, y_])
        y = Activation('relu', name='sub24_sum/relu')(y)

        aux_2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                       name='sub24_sum_interp')(y)
        y = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        y_ = Conv2D(128, 3, dilation_rate=2, name='conv_sub2')(y)
        y_ = BatchNormalization(name='conv_sub2_bn')(y_)

        # (1)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv1_sub1')(x)
        y = BatchNormalization(name='conv1_sub1_bn')(y)
        y = Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv2_sub1')(y)
        y = BatchNormalization(name='conv2_sub1_bn')(y)
        y = Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv3_sub1')(y)
        y = BatchNormalization(name='conv3_sub1_bn')(y)
        y = Conv2D(128, 1, name='conv3_sub1_proj')(y)
        y = BatchNormalization(name='conv3_sub1_proj_bn')(y)

        y = Add(name='sub12_sum')([y, y_])
        y = Activation('relu', name='sub12_sum/relu')(y)
        y = Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(x.shape[1]) * 2, int(x.shape[2]) * 2)),
                   name='sub12_sum_interp')(y)

        out = Conv2D(1, 1, activation='sigmoid', name='conv6_cls')(y)

        if mode == "training":
            aux_1 = Conv2D(1, 1, activation='sigmoid', name='sub4_out')(aux_1)
            aux_2 = Conv2D(1, 1, activation='sigmoid', name='sub24_out')(aux_2)

            # Losses
            aux_1_loss = KL.Lambda(lambda x: mask_loss_graph_16(*x), name="aux_1_loss")([input_gt_mask, aux_1])
            aux_2_loss = KL.Lambda(lambda x: mask_loss_graph_8(*x), name="aux_2_loss")([input_gt_mask, aux_2])
            out_loss = KL.Lambda(lambda x: mask_loss_graph_4(*x), name="out_loss")([input_gt_mask, out])

            # Model
            inputs = [input_image, input_gt_mask]
            outputs = [out, aux_1, aux_2, out_loss, aux_1_loss, aux_2_loss]
            model = KM.Model(inputs, outputs, name='ICNET')

        else:
            model = KM.Model(input_image, out, name='ICNET')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mhy.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mirror"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                                 'releases/download/v0.2/' \
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["out_loss", "aux_1_loss", "aux_2_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keep_dims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keep_dims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            # /path/to/logs/coco20171029T2315/mask_rcnn_mirror_0001.h5
            # regex = r".*/(\d{1})/mask\_rcnn\_\w+(\d{4})\.h5"
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, save_model_each_epoch=False):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gausssian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fcn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(decoder\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(decoder\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(decoder\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        # if self.epoch % 10 == 0:
        if save_model_each_epoch:
            callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                     histogram_freq=0, write_graph=True, write_images=False),
                         keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                         verbose=0, save_weights_only=True)
                         ]
        else:
            callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                     histogram_freq=0, write_graph=True, write_images=False)
                         ]
        # else:
        #     callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_dir,
        #                                              histogram_freq=0, write_graph=True, write_images=False)]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        windows = []
        # Actually only need handle one image.
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        windows = np.stack(windows)
        return molded_images, windows

    def unmold_detections(self, predict_mask):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # Convert neural network mask to full size mask
        final_mask = utils.unmold_mask(predict_mask)

        return final_mask

    def detect(self, imgname, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # print shape, min, max, and dtype
        if verbose:
            log("Processing images : {}".format(imgname))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        # images is a list which has only one image.
        molded_images, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        if verbose:
            log("molded_images", molded_images)
        # Run object detection
        predict_mask = self.keras_model.predict([molded_images], verbose=0)

        # Process detections
        results = []
        final_mask = self.unmold_detections(predict_mask)
        results.append({"mask": final_mask})

        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also retruned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE, \
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors : [anchor_count, (y1, x1, y2, x2)]
            # Sample anchor in original image size
            self.anchors = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            # self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(self.anchors, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and noramlized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image coordinates.
        [scale]  # size=1
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
