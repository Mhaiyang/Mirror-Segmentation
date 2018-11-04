640*640
lr = 1e-2

# decoder
resnet101.h5 batch 1*8  120epoch

pixel_accuracy       79.59
 
mean_iou             70.37

# fcn8
resnet101.h5 batch 1*8 120epoch

pixel_accuracy       74.45 

mean_iou             63.58

resnet101.h5 batch 4*6 80epoch

pixel_accuracy       80.07
 
mean_iou             67.56

mean_IOU             67.56
 
mean_ACC             87.59 

mean_BER             13.60

# psp
pspnet101_voc2012.h5 batch 4*1 40epoch

pixel_accuracy       63.92 

mean_iou             62.70

mean_IOU             62.70
 
mean_ACC             86.30 

mean_BER             18.28

# psp_edge_pooling
使用的是1/8的C5

pspnet101_voc2012.h5 batch 2*1 40epoch

pixel_accuracy       66.29 

mean_iou             64.69

# psp_edge_c1
使用的是C1来预测contour

pspnet101_voc2012.h5 batch 4*1 40epoch

pixel_accuracy       72.40
 
mean_iou             69.67

mean_IOU             69.67
 
mean_ACC             88.69 

mean_BER             14.50

# psp_edge_v2
在edge_c1的基础上，不叠加feature，edge branch只影响backbone。

pspnet101.voc2012.h5 batch 4*1 40 epoch

pixel_accuracy       61.26
 
mean_iou             60.07

# psp_edge_v3
没有semantic监督，验证semantic监督的作用（和edge_c1比）。同时验证concat的作用（和edge_v2比）。

pixel_accuracy       74.85
 
mean_iou             72.01

mean_IOU             72.01
 
mean_ACC             89.68 

mean_BER             13.18

# psp_edge_depth
loss mask:1  edge:10  depth:1e-4

在psp_edge_c1的基础上，增加了depth预测的分支，depth的输入为C2，depth有监督，叠加edge和depth的feature。

depth来自于Depth-Prediction，然后经过处理（将镜子的深度统一为平均值）得到。

pixel_accuracy       74.94 

mean_iou             71.64 

mean_psnr            19.58
 
mean_ssim            0.8674

mean_IOU             71.64
 
mean_ACC             89.42 

mean_BER             13.25

# psp_edge_depth_v2
在psp_edge_depth的基础上，用C1作为depth的输入

mean_IOU             74.21 

mean_ACC             90.32 

mean_BER             11.83 

# psp_edge_depth_v2_weighted
mean_IOU             69.36 

mean_ACC             88.80 

mean_BER             14.72 

mean_PSNR            19.43 

mean_SSIM            0.8641

# psp_edge_depth_v3
就是在v2的基础上，将edge分支和depth分支的卷积由三层变为6层，看看加深之后会不会变好
mean_IOU             72.67 

mean_ACC             90.16 

mean_BER             13.01 

mean_PSNR            19.43 

mean_SSIM            0.8706

# psp_depth
最原始的psp上增加depth分支，单独看看depth分支的作用

mean_IOU             71.75 

mean_ACC             89.69 

mean_BER             13.48 

mean_PSNR            19.22 

mean_SSIM            0.8609

# psp_dege_v3_no
在psp_edge_v3的基础上，没有edge监督，看看单纯的用一个skip connection和edge监督有什么区别。

mean_IOU             75.95
 
mean_ACC             90.99 

mean_BER             11.02

# psp_v2
去除了dropout

mean_IOU             70.90 

mean_ACC             89.04 

mean_BER             13.62

# psp_edge_depth_v4
复杂的两个分支
11.4

# psp_v3
就是最原版的psp，有dropout，看看之前的psp训练的有没有问题。


# mask rcnn
mean_IOU             78.01 

mean_ACC             88.17 

mean_BER             7.96