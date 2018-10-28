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

# edge
pspnet101_voc2012.h5 batch 2*1 40epoch

pixel_accuracy       66.29 

mean_iou             64.69

