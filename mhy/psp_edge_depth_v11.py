"""
  @Time    : 2018-9-1 05:14
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : mirror
  @File    : psp.py
  @Function: psp + edge + depth

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
import keras.losses as Kloss

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
def my_weighted_binary_crossentropy(target, output, weight, from_logits=False):
    """Weighted binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=weight)


def edge_loss_graph(input_gt_edge, edge):
    """Mask binary cross-entropy loss for the edge head.

    target_masks: [batch, height, width, 1]. bool. Convert it to
        a float32 tensor of values 0 or 1.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Compute binary cross entropy.
    # input_gt_edge = tf.expand_dims(input_gt_edge, -1)
    # gt_edge = tf.image.resize_bilinear(tf.cast(input_gt_edge, tf.float32), 80 * tf.ones(2, dtype=tf.int32))
    # gt_edge = tf.squeeze(gt_edge, -1)
    # gt_edge = tf.round(gt_edge)
    # P shape : [batch, channel]
    # weighted loss
    # P = tf.count_nonzero(input_gt_edge, [1, 2])
    # P = tf.cast(P, tf.float32)
    # total_number = 327680 * tf.ones(tf.shape(P), dtype=tf.float32)
    # N = total_number - P
    #
    # weight = N / P
    # weight = tf.squeeze(weight)
    # coefficient = P / total_number
    # coefficient = tf.squeeze(coefficient)

    gt_edge = K.cast(input_gt_edge, tf.float32)

    edge = K.squeeze(edge, -1)

    loss = K.binary_crossentropy(target=gt_edge, output=edge)
    # loss = my_weighted_binary_crossentropy(target=gt_edge, output=edge, weight=weight)
    # loss = coefficient * loss
    loss = K.mean(loss)

    return loss


def depth_loss_graph(input_gt_depth, depth):
    """Mask mean square error loss for the depth head.

    input_gt_depth: [batch, height, width]. tf.uint8. convert it to tf.float32
    depth: [batch, height, width, 1] float32 tensor. Generated from depth branch.
    """
    # Compute mse.
    gt_depth = K.cast(input_gt_depth, tf.float32)
    depth = K.squeeze(depth, -1)

    loss = Kloss.mean_absolute_error(y_true=gt_depth, y_pred=depth)

    return loss


def mask_loss_graph(input_gt_mask, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, height, width, 1]. bool. Convert it to
        a float32 tensor of values 0 or 1.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    # gt_mask = tf.image.resize_bilinear(tf.cast(input_gt_mask, tf.float32), 640 * tf.ones(2, dtype=tf.int32))
    # gt_mask = tf.round(gt_mask)
    # weighted loss
    # P = tf.count_nonzero(input_gt_mask, [1, 2])
    # P = tf.cast(P, tf.float32)
    # total_number = 327680 * tf.ones(tf.shape(P), dtype=tf.float32)
    # N = total_number - P
    #
    # weight = N / P
    # weight = tf.squeeze(weight)
    # coefficient = P / total_number
    # coefficient = tf.squeeze(coefficient)

    target_masks = K.cast(input_gt_mask, tf.float32)

    pred_masks = K.squeeze(pred_masks, -1)

    loss = K.binary_crossentropy(target=target_masks, output=pred_masks)
    # loss = my_weighted_binary_crossentropy(target=target_masks, output=pred_masks, weight=weight)
    # loss = coefficient * loss
    loss = K.mean(loss)

    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False):
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
    edge = dataset.load_edge(image_id)
    depth = dataset.load_depth(image_id)
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)
    mask = np.round(mask)

    edge = utils.resize_mask(edge, scale, padding, crop)
    edge = np.round(edge)

    depth = utils.pad_depth(depth, padding)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        k = random.randint(0, 1, 2, 3)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)

    return image, mask, edge, depth


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   batch_size=1):
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
            image, gt_masks, gt_edge, gt_depth = load_image_gt(dataset, config, image_id, augment=augment)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1]), dtype=gt_masks.dtype)
                batch_gt_edge = np.zeros(
                    (batch_size, gt_edge.shape[0], gt_edge.shape[1]), dtype=gt_edge.dtype)
                batch_gt_depth = np.zeros(
                    (batch_size, gt_depth.shape[0], gt_depth.shape[1]), dtype=gt_depth.dtype)

            # Add to batch
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_masks[b] = gt_masks
            batch_gt_edge[b] = gt_edge
            batch_gt_depth[b] = gt_depth

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_masks, batch_gt_edge, batch_gt_depth]
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
#  Resnet for PSPNet
############################################################

def BN(name=""):
    return KL.BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


class Interp(KL.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = tf.image.resize_images(inputs, [new_height, new_width],
                                         align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = KL.Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                         use_bias=False)(prev)
    elif modify_stride is True:
        prev = KL.Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                         use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = KL.Activation('relu')(prev)

    prev = KL.ZeroPadding2D(padding=(pad, pad))(prev)
    prev = KL.Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                     name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = KL.Activation('relu')(prev)
    prev = KL.Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                     use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = KL.Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                         use_bias=False)(prev)
    elif modify_stride is True:
        prev = KL.Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                         use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = KL.Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = KL.Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = KL.Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = KL.Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = KL.Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                     use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = KL.Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = KL.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                     use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = KL.Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = KL.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                     use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = KL.Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    C1 = relu1

    res = KL.MaxPooling2D(pool_size=(3, 3), padding='same',
                          strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    C2 = res

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)

    C3 = res

    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    C4 = res

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = KL.Activation('relu')(res)
    return res, C1, C2, C3, C4


# ##############33 kernel module #######################33
def interp_block(prev_layer, level, feature_map_shape, input_shape, branch):
    if input_shape == [640, 640]:
        kernel_strides_map = {1: 80,
                              2: 40,
                              4: 20,
                              8: 10}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + branch + "_conv",
        "conv5_3_pool" + str(level) + branch + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = KL.AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = KL.Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                           use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = KL.Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape, branch="_semantic"):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(math.ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape, branch)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape, branch)
    interp_block4 = interp_block(res, 4, feature_map_size, input_shape, branch)
    interp_block8 = interp_block(res, 8, feature_map_size, input_shape, branch)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = KL.Concatenate(axis=3)([res, interp_block8, interp_block4, interp_block2, interp_block1])
    return res


############################################################
#  Network Class
############################################################

class PSP_EDGE_DEPTH(object):
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
            # 2. GT Edge [batch, height, width]
            input_gt_edge = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]], name="input_gt_edge", dtype=tf.uint8)
            # 3. GT Depth [batch, height, width]
            input_gt_depth = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]], name="input_gt_depth", dtype=tf.float32)

        # Build the backbone layers.
        res, C1, C2, C3, C4 = ResNet(input_image, layers=101)

        # depth branch. 1/2 of input image
        depth_c1 = KL.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="depth_c1", use_bias=False)(C1)
        depth_c1 = BN(name="depth_c1_bn")(depth_c1)
        depth_c1 = KL.Activation("relu")(depth_c1)

        depth_c2 = KL.UpSampling2D(size=(2, 2))(C2)
        depth_c2 = KL.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="depth_c2", use_bias=False)(depth_c2)
        depth_c2 = BN(name="depth_c2_bn")(depth_c2)
        depth_c2 = KL.Activation("relu")(depth_c2)

        depth_c3 = KL.UpSampling2D(size=(4, 4))(C3)
        depth_c3 = KL.Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="depth_c3", use_bias=False)(depth_c3)
        depth_c3 = BN(name="depth_c3_bn")(depth_c3)
        depth_c3 = KL.Activation("relu")(depth_c3)

        depth_c123 = KL.Concatenate(axis=3)([depth_c1, depth_c2, depth_c3])
        depth_c123 = KL.Conv2D(64, (3, 3), padding="same", name="depth_c123", use_bias=False)(depth_c123)
        depth_c123 = BN(name="depth_c123_bn")(depth_c123)
        depth_c123 = KL.Activation("relu")(depth_c123)

        depth_c4 = KL.UpSampling2D(size=(2, 2))(C4)
        depth_c4 = KL.Conv2D(128, (3, 3), padding="same", name="depth_c4_1", use_bias=False)(depth_c4)
        depth_c4 = BN(name="depth_c4_bn1")(depth_c4)
        depth_c4 = KL.Activation("relu")(depth_c4)

        depth_c4 = KL.UpSampling2D(size=(2, 2))(depth_c4)
        depth_c4 = KL.Conv2D(64, (3, 3), padding="same", name="depth_c4_2", use_bias=False)(depth_c4)
        depth_c4 = BN(name="depth_c4_bn2")(depth_c4)
        depth_c4 = KL.Activation("relu")(depth_c4)

        depth_fusion = KL.Concatenate(axis=3)([depth_c123, depth_c4])
        depth_fusion = KL.Conv2D(128, (3, 3), padding="same", name="depth_fusion_conv1", use_bias=False)(depth_fusion)
        depth_fusion = BN(name="depth_fusion_conv1_bn")(depth_fusion)
        depth_fusion = KL.Activation("relu")(depth_fusion)

        depth_feature = KL.AveragePooling2D(pool_size=(4, 4))(depth_fusion)
        depth_feature = KL.Conv2D(512, (3, 3), padding="same", name="depth_feature", use_bias=False)(depth_feature)
        depth_feature = BN(name="depth_deature_bn")(depth_feature)
        depth_feature = KL.Activation("relu")(depth_feature)

        depth = KL.Conv2DTranspose(1, (3, 3), strides=2, padding="same", name="middle_depth")(depth_fusion)

        # semantic branch
        res = KL.Conv2D(1536, (3, 3), padding="same", name="C5_conv_to_1536", use_bias=False)(res)
        res = BN(name="C5_conv_to_1536_bn")(res)
        res = KL.Activation("relu")(res)

        res = KL.Concatenate(axis=3)([res, depth_feature])
        res = BN(name="before_psp_bn")(res)
        res = KL.Activation("relu")(res)

        psp_semantic = build_pyramid_pooling_module(res, [640, 640], branch="_semantic")
        psp_semantic = BN(name="psp_semantic_bn")(psp_semantic)
        psp_semantic = KL.Activation("relu")(psp_semantic)

        x = KL.Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="semantic_conv", use_bias=False)(psp_semantic)
        x = BN(name="after_psp_bn")(x)
        x = KL.Activation('relu')(x)

        # edge branch. 1/4 of input image
        edge_c1 = KL.Conv2D(256, (3, 3), strides=(2, 2), padding="same", name="edge_c1", use_bias=False)(C1)
        edge_c1 = BN(name="edge_c1_bn")(edge_c1)
        edge_c1 = KL.Activation("relu")(edge_c1)

        edge_c2 = KL.Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="edge_c2", use_bias=False)(C2)
        edge_c2 = BN(name="edge_c2_bn")(edge_c2)
        edge_c2 = KL.Activation("relu")(edge_c2)

        edge_c3 = KL.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", name="edge_c3", use_bias=False)(C3)
        edge_c3 = BN(name="edge_c3_bn")(edge_c3)
        edge_c3 = KL.Activation("relu")(edge_c3)

        edge_c123 = KL.Concatenate(axis=3)([edge_c1, edge_c2, edge_c3])
        edge_c123 = KL.Conv2D(256, (3, 3), padding="same", name="edge_c123_conv1", use_bias=False)(edge_c123)
        edge_c123 = BN(name="edge_c123_conv1_bn")(edge_c123)
        edge_c123 = KL.Activation("relu")(edge_c123)
        edge_c123 = KL.Conv2D(256, (3, 3), padding="same", name="edge_c123_conv2", use_bias=False)(edge_c123)
        edge_c123 = BN(name="edge_c123_conv2_bn")(edge_c123)
        edge_c123 = KL.Activation("relu")(edge_c123)

        edge_feature = KL.AveragePooling2D(pool_size=(2, 2))(edge_c123)
        edge_feature = KL.Conv2D(256, (3, 3), padding="same", name="edge_feature", use_bias=False)(edge_feature)
        edge_feature = BN(name="edge_deature_bn")(edge_feature)
        edge_feature = KL.Activation("relu")(edge_feature)

        edge = KL.Conv2D(1, (3, 3), strides=(1, 1), padding="same", name="middle_edge")(edge_c123)
        edge = Interp([640, 640])(edge)
        edge = KL.Activation("sigmoid")(edge)

        # final fusion
        m = KL.Concatenate(axis=3, name="fusion")([x, edge_feature])
        m = KL.Conv2D(1, (3, 3), padding="same", name="final_conv")(m)
        m = Interp([640, 640])(m)
        predict_mask = KL.Activation('sigmoid')(m)

        if mode == "training":
            # middle loss
            edge_loss = KL.Lambda(lambda x: edge_loss_graph(*x),
                                  name="edge_loss")([input_gt_edge, edge])
            depth_loss = KL.Lambda(lambda x: depth_loss_graph(*x),
                                   name="depth_loss")([input_gt_depth, depth])
            # final loss
            mask_loss = KL.Lambda(lambda x: mask_loss_graph(*x), name="mask_loss")([input_gt_mask, predict_mask])

            # Model
            inputs = [input_image, input_gt_mask, input_gt_edge, input_gt_depth]
            outputs = [predict_mask, edge_loss, depth_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='PSP_EDGE_DEPTH')
            model.load_weights(self.config.Pretrained_Model_Path, by_name=True)

        else:
            model = KM.Model(input_image, [predict_mask, edge, depth], name='PSP_EDGE_DEPTH')

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
        loss_names = ["mask_loss", "edge_loss", "depth_loss"]
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
            "heads": r"(conv6\_.*)",
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
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

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
        predict_mask, edge, depth = self.keras_model.predict([molded_images], verbose=0)

        # Process detections
        results = []
        final_mask = self.unmold_detections(predict_mask)
        final_edge = utils.unmold_edge(edge)
        final_depth = utils.unmold_depth(depth)
        results.append({"mask": final_mask, "edge": final_edge, "depth": final_depth})

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

