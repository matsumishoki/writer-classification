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

image = plt.imread(os.path.join(change_path, "0001-1-cropped.png"))

# 画像のy(縦座標)座標の総数を取得する
image_y = image.shape[0]

# 画像のx座標(横座標)の総数を取得する
image_x = image.shape[1]

# 画像サイズを指定する
image_size = 200
cut_image_edge_size = 200
# xとy共に200ずつの間隔で座標を取得する
y_start_point = range(0, image_y, image_size)
x_start_point = range(0, image_x, image_size)

num_y_start_point = len(y_start_point)
num_x_start_point = len(x_start_point)
print "num_y_start_point:", num_y_start_point
print "num_x_start_point:", num_x_start_point

# file数分のループを回す

# 訓練データを生成する
# 横座標を固定で縦座標を横にズラしていき，200*200の画像を取得する
# 端の余った画像部分は使用しない
for x in range(num_x_start_point-1):
    x_point = x_start_point[x]
    for y in range(num_y_start_point-1):
        y_point = y_start_point[y]
        print "train_x_point:", x_point
        print "train_y_point:", y_point
        train_cropped_image = image[y_point:y_point+200, x_point:x_point+200]
#        plt.imshow(cropped_image, cmap=plt.cm.gray)
#        plt.show()
#        plt.draw()

# テストデータを生成する
# 横座標をランダムに選び，選んだ位置から縦に200横に200の画像サイズを指定し，画像を切り取る
# 切り取る時に縦200，横200の画像サイズが切り取れる位置を指定できるようにする
x_select_points = image_x - cut_image_edge_size
y_select_points = image_y - cut_image_edge_size
x_select_point = np.random.permutation(x_select_points)
y_select_point = np.random.permutation(y_select_points)

test_cropped_image = image[y_select_point[0]:y_select_point[0]+200,
                           x_select_point[0]:x_select_point[0]+200]
print "test_x_point:", x_select_point[0]
print "test_y_point:", y_select_point[0]
plt.imshow(test_cropped_image, cmap=plt.cm.gray)
