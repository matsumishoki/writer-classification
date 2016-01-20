# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:10:53 2016

@author: matsumi
"""
import numpy as np
import scipy as sp
from xml.etree import ElementTree
import os
import skimage
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 検索するデータセットのファイルのtop_pathを指定する
top_path = os.path.join("CVL_ConvNet_data")
temp_list = [data_filepath for data_filepath in os.walk(top_path)]
num_temp_list = len(temp_list)

tup = temp_list[0]
(dirpath, dirnames, filenames) = tup

num_tup = len(tup)
num_dirpath = len(dirpath)
num_dir = len(dirnames)
num_file = len(filenames)

change_path = os.path.join("CVL_ConvNet_data")

cut_image = plt.imread(os.path.join(change_path, "0001-1-cropped.png"))

# 画像のy(縦座標)座標の総数を取得する
cut_image_y = cut_image.shape[0]

# 画像のx座標(横座標)の総数を取得する
cut_image_x = cut_image.shape[1]

# xとy共に200ずつの間隔で座標を取得する
y_start_point = range(0, cut_image_y, 200)
x_start_point = range(0, cut_image_x, 200)

num_y_start_point = len(y_start_point)
num_x_start_point = len(x_start_point)
print "num_y_start_point:", num_y_start_point
print "num_x_start_point:", num_x_start_point

# 横座標を固定で縦座標を変化させて200*200の画像を取得する
# 端の余った画像部分は使用しない
for x in range(num_x_start_point-1):
    print "a"
    print x
    x_point = x_start_point[x]
    for y in range(num_y_start_point-1):
        print "b"
        y_point = y_start_point[y]
        plt.imshow(cut_image[x_point:x_point+200, :200], cmap=plt.cm.gray)
# plt.imshow(cut_image[:200, :200], cmap=plt.cm.gray)
