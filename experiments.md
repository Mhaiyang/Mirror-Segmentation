640*640
lr = 1e-2

# decoder
resnet101.h5 batch 1*8

pixel_accuracy       79.52
 
mean_iou             70.30

# fcn8
resnet101.h5 batch 1*8

pixel_accuracy       74.49 
mean_iou             63.54

# psp
pspnet101_voc2012.h5 batch:2*1
