# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:27:12 2016

@author: matsumi
"""
import os
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import copy
import chainer.functions as F
from chainer import Variable, FunctionSet
from chainer.optimizers import SGD, Adam


def error_and_accuracy(w_1, w_2, b_1, b_2, x_data, t_data):
    x = Variable(x_data)
    t = Variable(t_data)

    a_z = F.linear(x, w_1, b_1)
    z = F.tanh(a_z)
    a_y = F.linear(z, w_2, b_2)

    error = F.softmax_cross_entropy(a_y, t)
    accuracy = F.accuracy(a_y, t)
    return error.data, accuracy.data * 100

if __name__ == '__main__':
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
    for filename in filenames:
        loadFileFullpath = os.path.join(dirpath, filename)
        print loadFileFullpath
        image = plt.imread(os.path.join(dirpath, filename))

        # 画像データの名前と拡張子を分離する
        name, ext = os.path.splitext(filename)

        # テスト画像だけを指定し，ファイルに保存する
        text_type = name[5:6]
        if(int(text_type) > 3):
            print "filename:", filename
            continue
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        plt.draw()
