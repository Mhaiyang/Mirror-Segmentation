import os
import re
import shutil
import numpy as np
import skimage.io

data_path = "/home/iccd/Desktop/data_json_old"
output_path = "/home/iccd/Desktop/MSD_json"

for line in open("./list.txt"):
    print(line)
    imglist = os.listdir(data_path + "/image")
    regex = re.compile(r"^%s_.*"%line[:-1])
    print(regex)
    for string in imglist:
        name = re.findall(regex, string)
        if name:
            break
    name = name[0]
    print(name)

    shutil.move(data_path + "/image/" + name, output_path + "/test_disorder/image/" + name)
    # shutil.move(data_path + "/mask/" + name[:-4] + ".png", output_path + "/test_noorder/mask/" + name[:-4] + ".png")
    shutil.move(data_path + "/mask/" + name[:-4] + "_json", output_path + "/test_disorder/mask/" + name[:-4] + "_json")

