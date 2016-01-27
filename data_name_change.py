# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 04:14:24 2016

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
top_path = os.path.join("resized_normal")
temp_list = [data_filepath for data_filepath in os.walk(top_path)]
num_temp_list = len(temp_list)


tup = temp_list[0]
(dirpath, dirnames, filenames) = tup

for filename in filenames:
    loadFileFullpath = os.path.join(dirpath, filename)
    image = plt.imread(os.path.join(dirpath, filename))
#    print loadFileFullpath

    # 画像データの名前と拡張子を分離する
    name, ext = os.path.splitext(filename)
    print name[3:]
    name = name[3:]

    save_path = "C:\\Users\\matsumi\\Desktop\\writer classification\\resized_test\\"
    saveFilename = save_path + name + ".png"   # name + 新しい拡張子(.png)
    print saveFilename
    skimage.io.imsave(saveFilename, image)
