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

mean_IOU             71.64 
mean_ACC_all         89.84 
mean_ACC_mirror      84.42 
mean_BER             11.13

# psp
pspnet101_voc2012.h5 batch 4*1 40epoch

pixel_accuracy       63.92 

mean_iou             62.70

mean_IOU             62.70
 
mean_ACC             86.30 

mean_BER             18.28

第二次训练：psp_v3

mean_IOU             68.96 

mean_ACC             88.49 

mean_BER             14.76

mean_IOU             76.66 
mean_ACC_all         91.71 
mean_ACC_mirror      79.52 
mean_BER             10.80

第三次训练：psp_v4

mean_IOU             65.98
 
mean_ACC             87.53 

mean_BER             16.51

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

# psp_v3
就是最原版的psp，有dropout，看看之前的psp训练的有没有问题。
居然结果比之前训练的高。

# psp_edge_depth_v4
复杂的两个分支
11.4 跑到一半应该是显存不够了。

mean_IOU             75.45 

mean_ACC             91.04 

mean_BER             11.33 

mean_PSNR            20.00 

mean_SSIM            0.8650

# psp_edge_depth_v5
在v4的基础上，换了最新的depth的gt，并用ln损失代替了mse，也去掉了dropout，算是最好的一个版本，希望结果最好。
11.4 
1080Ti和服务器上都跑了。
1080Ti是1:10:10 batch 2
服务器是1:1:1 batch 4

1080Ti:

mean_IOU             72.94
 
mean_ACC             90.06 

mean_BER             12.46 

mean_MSE             0.08

服务器：

mean_IOU             73.77
 
mean_ACC             90.32 

mean_BER             12.06 

mean_MSE             7.14

# psp_edge_depth_v6

重新设计的pipeline，有大的skip connection。

mean_IOU             56.84 

mean_ACC             84.73 

mean_BER             21.24 

# depth

看看能不能学镜子区域平均后的深度图。

结论是学不出来，大的镜子根本不能学到平均值。

# psp_edge_depth_v7
在v4的基础上，直接学原始的depth。1e-3.

不好，跟v4差很多。

# psp_edge_depth_v8
在v7的基础上，将c1，c2，c3，c4都叠加在c5上，然后送入psp。1e-3.

啥都没学出来。

# unet
80 epoch

61.46
86.12
17.81

mean_IOU             63.63 
mean_ACC_all         87.33 
mean_ACC_mirror      72.08 
mean_BER             16.55

# psp_edge_depth_v9
算是在v7的基础上， edge分支不变， depth分支采用了arxiv18的，学习率0.01，比例1:1:1。depth是0-1之间的，L1损失函数。

mean_IOU             71.34 

mean_ACC             89.67 

mean_BER             13.33 

mean_MSE             1842.60

# segnet
lr=0.01

mean_IOU             71.88 

mean_ACC             89.93 

mean_BER             12.62

mean_IOU             74.15 
mean_ACC_all         91.10 
mean_ACC_mirror      81.00 
mean_BER             11.35

# psp_edge_depth_v10
在v9的基础上，比例1:10:0.1，融合换成了pooling之后再融合。edge分支稍微变复杂了。

mean_IOU             66.57 

mean_ACC             88.19 

mean_BER             16.13

# psp_edge_depth_v11
在v10的基础上，为了显存，edge分支简化了，和v9一样，将depth融合放到psp module之前。

1:1:1

mean_IOU             69.13
 
mean_ACC             88.75 

mean_BER             14.69

# psp_edge_depth_v12
在v4的基础上，将edge和depth都叠加在了psp module之前。

mean_IOU             75.72 

mean_ACC             91.45 

mean_BER             11.09

# psp_edge_depth_v13
depth分支采用arxiv，edge分支只有三层，都叠加在psp之前，比例1:0.5:0.5.
11.10

# psp_edge_depth_v14
mean_IOU             83.43 
mean_ACC_all         94.25 
mean_ACC_mirror      89.27 
mean_BER             6.85

# psp_edge_depth_v14_psp_depth
mean_IOU             80.46 
mean_ACC_all         93.09 
mean_ACC_mirror      85.69 
mean_BER             8.32

# psp_edge_depth_v14_psp_edge
mean_IOU             80.09 
mean_ACC_all         93.13 
mean_ACC_mirror      84.19 
mean_BER             8.79




# ICNet ECCV2018

mean_IOU             70.42
 
mean_ACC             89.34 

mean_BER             12.94

120 epoch
mean_IOU             72.68 
mean_ACC_all         90.52 
mean_ACC_mirror      81.96 
mean_BER             11.62

150epoch
mean_IOU             72.40 
mean_ACC_all         90.40 
mean_ACC_mirror      81.79 
mean_BER             11.71

# mask rcnn
mean_IOU             78.01  79.81

mean_ACC             88.17  92.55

mean_BER             7.96  9.46

mean_IOU             81.53 
mean_ACC_all         93.50 
mean_ACC_mirror      86.48 
mean_BER             8.50

# PiCANet(CVPR2018)
mean_IOU             93.18
 
mean_ACC_all         98.06 

mean_ACC_mirror      96.50 

mean_BER             2.49