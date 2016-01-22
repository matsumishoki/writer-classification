# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:38:56 2016

@author: matsumi
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from xml.etree import ElementTree
import os
import skimage
import skimage.io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# 検索するデータセットのファイルのtop_pathを指定する
top_path = os.path.join("CVL", "cvl-database-cropped-1-1",
                        "cvl-database-cropped-1-1")
temp_list = [data_filepath for data_filepath in os.walk(top_path)]
num_temp_list = len(temp_list)
print"num_temp_list:", num_temp_list

tup = temp_list[0]
(dirpath, dirnames, filenames) = tup

num_tup = len(tup)
num_dirpath = len(dirpath)
num_dir = len(dirnames)
num_file = len(filenames)

print "num_tup:", num_tup
print "num_dirpath:", num_dirpath
print "num_dir:", num_dir
print "num_file:", num_file

# グレースケール化し，二値化した後のファイルのtop_pathを指定する
change_path = os.path.join("CVL_ConvNet_data")

a = plt.imread(os.path.join(change_path, "0001-1-cropped.tif"))
y = a.shape[0]
x = a.shape[1]
l = a[50:(x-50), 400:(y-400)]
plt.imshow(l)

for filename in filenames:
    loadFileFullpath = os.path.join(dirpath, filename)
    print loadFileFullpath

    # グレースケール化した後に二値化する
    image = plt.imread(os.path.join(dirpath, filename))
    image_gray = rgb2gray(image)
    thresh = threshold_otsu(image_gray)
    image_binary = image_gray > thresh
    image_inv = np.logical_not(image_binary)
    image_data = np.uint8(image_inv * 255)

    # 画像を横(左400，右400)，縦(上50,下50)切り取る
    image_data = image_data[50:(x-50), 400:(y-400)]

    # 画像データの名前と拡張子を分離する
    name, ext = os.path.splitext(filename)

    #文章によって使うファイルと使わないファイルを分ける
    text_type = name[5:6]
    if(int(text_type) > 4):
        continue

    save_path = "C:\\Users\\matsumi\\Desktop\\writer classification\\CVL_ConvNet_data\\"
    saveFilename = save_path + name + ".png"   # name + 新しい拡張子(.png)
    print saveFilename
    skimage.io.imsave(saveFilename, image_data)
