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


def loss_and_accuracy(model, x_data, t_data, train=False):
    x = Variable(x_data.reshape(-1, 1, 28, 28))
    t = Variable(t_data)

    # 順伝播
    h = model.conv_11(x)
    h = model.conv_12(h)
#    h = model.conv_13(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    h = model.conv_2(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    h = model.conv_3(h)
    h = F.relu(h)
#    h = F.dropout(h, ratio=0.9, train=train)
    h = model.linear_1(h)
    h = F.relu(h)
#    h = F.dropout(h, ratio=0.9, train=train)
    a_y = model.linear_2(h)

    loss = F.softmax_cross_entropy(a_y, t)
    accuracy = F.accuracy(a_y, t)

    return loss, cuda.to_cpu(accuracy.data) * 100

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

        # 3種類の訓練画像だけを指定する
        text_type = name[5:6]
        if(int(text_type) > 3):
            print "filename:", filename
            continue
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        plt.draw()
        image_y = image.shape[0]
        image_x = image.shape[1]
            # １人あたり3種類の画像から1枚ずつ切り出し画像を作成する

    # 超パラメータの定義
    learning_rate = 0.000001  # learning_rate(学習率)を定義する
    max_iteration = 1000      # 学習させる回数
    batch_size = 200       # ミニバッチ1つあたりのサンプル数
    dim_hidden_1 = 500         # 隠れ層の次元数を定義する
    dim_hidden_2 = 500
    wscale_1 = 1.0
    wscale_2 = 1.0
    wscale_3 = 1.0
    l_2 = 0.0015

    # 学習させるループ

        # mini batchi SGDで重みを更新させるループ

            # 逆伝播

        # 誤差

        # 学習曲線をプロットする

    # 学習済みのモデルをテストセットで誤差と正解率を求める

    # wの可視化
