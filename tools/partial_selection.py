import os
import shutil
import random

rate = 0.05
image_path = "/home/iccd/data/types/16/"
select_to_test = "/home/iccd/data/MSD7/test/image/"
select_to_train = "/home/iccd/data/MSD7/train/image/"

imglist = os.listdir(image_path)
print(imglist)
random.shuffle(imglist)
print(imglist)
print(len(imglist))

test_number = int(len(imglist)*rate)
print("test number is : {}".format(test_number))

for i, imgname in enumerate(imglist):
    name = imgname.split(".")[0]

    if i < test_number:
        shutil.copy(image_path + imgname, select_to_test)
        print("{} is selected to test".format(imgname))
    else:
        shutil.copy(image_path + imgname, select_to_train)
        print("{} is selected to train".format(imgname))


print("test number is : {}".format(test_number))
print("train number is : {}".format(len(imglist) - test_number))
print("ok")