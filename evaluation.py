'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.

n_cl : number of classes included in ground truth segmentation

n_ij : number of pixels of class i predicted to belong to class j

t_i : total number of pixels of class i in ground truth segmentation

'''

import numpy as np
import os
import yaml
from PIL import Image
import skimage.transform

def get_mask(imgname, MASK_DIR):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = MASK_DIR
    mask_path = mask_folder + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    width, height = mask.size
    num_obj = np.max(mask)

    gt_mask = np.zeros([height, width, 1], dtype=np.uint8)
    for index in range(num_obj):
        """j is row and i is column"""
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i, j))
                if at_pixel == index + 1:
                    gt_mask[j, i, 0] = 1
    return gt_mask

def resize_mask(gt_mask, size):

    resized_gt_mask = skimage.transform.resize(gt_mask, size, order=1, mode="constant")
    resized_gt_mask = np.where(resized_gt_mask == 0, 0, 1).astype(np.bool)

    return resized_gt_mask


def pixel_accuracy(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    sum_n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    sum_t_i = np.sum(gt_mask)

    if sum_t_i == 0:
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def IoU(predict_mask, gt_mask):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Here, n_cl = 1 as we have only one class (mirror).
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        IoU = 0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    IoU = n_ii / (t_i + n_ij - n_ii)

    return IoU


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_



def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

