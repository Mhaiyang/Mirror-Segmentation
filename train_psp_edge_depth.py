"""
  @Time    : 2018-8-29 00:00
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : mirror segmentation
  @File    : train.py
  @Function: train code.

"""
import os
import mirror
import mhy.psp_edge_depth_v14_psp_edge as modellib

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "log", "psp_edge_depth_v14_psp_edge")
    
config = mirror.MirrorConfig()
config.display()

# Configuration
dataset_root_path = os.path.abspath(os.path.join(ROOT_DIR, "./data_640"))
train_folder = dataset_root_path + "/train"
val_folder = dataset_root_path + "/val"
train_image_folder = train_folder + "/image"
train_mask_folder = train_folder + "/mask"
val_image_folder = val_folder + "/image"
val_mask_folder = val_folder + "/mask"
train_imglist = os.listdir(train_image_folder)
train_count = len(train_imglist)
val_imglist = os.listdir(val_image_folder)
val_count = len(val_imglist)
print("Train Image Count : {} \nValidation Image Count : {}".format(train_count, val_count))

# Training dataset
dataset_train = mirror.MirrorDataset()
dataset_train.load_mirror(train_count, train_image_folder,
                          train_mask_folder, train_imglist)     # add class and add image.
dataset_train.prepare("train")

# Validation dataset
dataset_val = mirror.MirrorDataset()
dataset_val.load_mirror(val_count, val_image_folder,
                        val_mask_folder, val_imglist)      # add class and add image
dataset_val.prepare("validation")

# ## Create Model  ###
model = modellib.PSP_EDGE_DEPTH(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "resnet101"  # resnet101 or last

if init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# ## Training
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=35,
            layers='all',
            save_model_each_epoch=False)
model_path = os.path.join(MODEL_DIR, "mirror_psp_edge_depth_v14_psp_edge_all_35.h5")
model.keras_model.save_weights(model_path)

# save model weights.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=45,
            layers='all',
            save_model_each_epoch=False)
model_path = os.path.join(MODEL_DIR, "mirror_psp_edge_depth_v14_psp_edge_all_45.h5")
model.keras_model.save_weights(model_path)

