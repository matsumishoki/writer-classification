# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:21:32 2016

@author: matsumi
"""
import numpy as np
import scipy as sp
from xml.etree import ElementTree
import os
import skimage
import skimage.io
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

for filename in filenames:
    loadFileFullpath = os.path.join(dirpath, filename)
    print loadFileFullpath
    image = plt.imread(os.path.join(dirpath, filename))

    # 画像データの名前と拡張子を分離する
    name, ext = os.path.splitext(filename)

    # テスト画像だけを指定し，ファイルに保存する
    text_type = name[5:6]
    if(int(text_type) < 4):
        continue
    save_path = "C:\\Users\\matsumi\\Desktop\\writer classification\\CVL_test_data\\"
    saveFilename = save_path + name + ".png"   # name + 新しい拡張子(.png)
    print saveFilename
    skimage.io.imsave(saveFilename, image)
