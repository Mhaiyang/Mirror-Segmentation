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

# psp
pspnet101_voc2012.h5 batch 4*1 40epoch

pixel_accuracy       63.92 

mean_iou             62.70

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

# psp_edge_v2
在edge_c1的基础上，不叠加feature，edge branch只影响backbone。

pspnet101.voc2012.h5 batch 4*1 40 epoch

pixel_accuracy       61.26
 
mean_iou             60.07

# psp_edge_v3
没有semantic监督，验证semantic监督的作用（和edge_c1比）。同时验证concat的作用（和edge_v2比）。

# psp_edge_depth
在psp_edge_c1的基础上，增加了depth预测的分支，depth的输入为C2，depth有监督，叠加edge和depth的feature。

depth来自于Depth-Prediction，然后经过处理（将镜子的深度统一为平均值）得到。