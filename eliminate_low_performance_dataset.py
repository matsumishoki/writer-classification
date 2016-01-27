# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:08:27 2016

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
top_path = os.path.join("CVL_test_data")
temp_list = [data_filepath for data_filepath in os.walk(top_path)]
tup = temp_list[0]
(dirpath, dirnames, filenames) = tup

image_size = 200
lower_text = 4000
images = []
file_numbers = []
for filename in filenames:
    image = plt.imread(os.path.join(dirpath, filename))

    # 画像データの名前と拡張子を分離する
    name, ext = os.path.splitext(filename)

    # 3種類の訓練画像だけを指定する
    heigh = image.shape[0]
    width = image.shape[1]

    # １人あたり3種類の画像から1枚ずつ切り出し画像を作成する
    x_select_points = width - image_size
    y_select_points = heigh - image_size
    x_select_point = np.random.permutation(x_select_points)
    y_select_point = np.random.permutation(y_select_points)
    y_p = y_select_point[0]
    x_p = x_select_point[0]

    image_1 = image[y_p:y_p+image_size, x_p:x_p+image_size]

    # 切り出し画像の切り出し位置を変更をする

    # x_pの位置を変更してループを回す

        # y_pの位置を変更してループを回す

            # if num_text_range < lower_textが4000を超えるとループを抜けて

            # 外側のループ(for filename in filenames:)に戻る

            # それ以外ならば，y_pの位置を変更する．y_p全通り見たならばx_pを変更する

    # 1回だけ変更する←これを上のスケッチのようにしたい
    num_text_range = np.sum(np.ones((image_size, image_size)) == image)
    if num_text_range < lower_text:
        y_p = y_select_point[1]
        x_p = x_select_point[1]

        image_2 = image[y_p:y_p+image_size, x_p:x_p+image_size]
    text_name = name[:4]
    plt.imshow(image_1, cmap=plt.cm.gray)
    plt.show()
    plt.draw()
    plt.imshow(image_2, cmap=plt.cm.gray)
    plt.show()
    plt.draw()
#    print image
