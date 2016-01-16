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

a = plt.imread(os.path.join(change_path, "0001-1-cropped.png"))
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

    # 画像を横(左400，右400)，縦(上50,下50)切り取る
    image_cut_y = image_binary[0].shape
    image_cut_x = image_binary[1].shape
    image_data = image_binary[50:(x-50), 400:(y-400)]
    # 画像データの名前と拡張子を分離する

    name, ext = os.path.splitext(filename)
    saveFilename = "C:\\Users\\matsumi\\Desktop\\CVL_ConvNet_data\\" + name + ".png"   # name + 新しい拡張子(.png)
    print saveFilename
    plt.imsave(saveFilename, image_data, cmap=plt.cm.gray)



#for filename in filenames:
#    loadFileFullpath = os.path.join(dirpath, filename)
#    print loadFileFullpath
#    # グレースケール化した後に二値化する
#    image = plt.imread(os.path.join(dirpath, filename))
#    image_gray = rgb2gray(image)
#    thresh = threshold_otsu(image_gray)
#    image_binary = image_gray > thresh
#
#    name, ext = os.path.splitext(filename)
#    saveFilename = "C:\\Users\\matsumi\\Desktop\\CVL_ConvNet\\" + name + ".png"   # name + 新しい拡張子(.png)
#    print saveFilename
#    plt.imsave(saveFilename, image_binary, cmap=plt.cm.gray)
