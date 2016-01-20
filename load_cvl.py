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

change_path = os.path.join("CVL_ConvNet_data")

cut_image = plt.imread(os.path.join(change_path, "0001-1-cropped.png"))
plt.imshow(cut_image[:200, :200], cmap=plt.cm.gray)
