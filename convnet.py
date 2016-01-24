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


# エポック毎に訓練データを作成する関数
def make_epoch_train_data():
    # 検索するデータセットのファイルのtop_pathを指定する
    top_path = os.path.join("CVL_ConvNet_data")
    temp_list = [data_filepath for data_filepath in os.walk(top_path)]
    tup = temp_list[0]
    (dirpath, dirnames, filenames) = tup

    image_size = 200
    images = []
    text_names = []
    for filename in filenames:
        image = plt.imread(os.path.join(dirpath, filename))

        # 画像データの名前と拡張子を分離する
        name, ext = os.path.splitext(filename)

        # 3種類の訓練画像だけを指定する
        text_type = name[5:6]
        if(int(text_type) > 3):
            continue
        heigh = image.shape[0]
        width = image.shape[1]

        # １人あたり3種類の画像から1枚ずつ切り出し画像を作成する
        x_select_points = width - image_size
        y_select_points = heigh - image_size
        x_select_point = np.random.permutation(x_select_points)
        y_select_point = np.random.permutation(y_select_points)
        y_p = y_select_point[0]
        x_p = x_select_point[0]

        image = image[y_p:y_p+image_size, x_p:x_p+image_size].copy()
        text_name = name[:4]
#        print text_name
        images.append(image)
        text_names.append(text_name)

    x = np.array(images).reshape(-1, 1, image_size, image_size)
    t = np.array(text_names)
    t = t.astype(np.int32)
    return x, t


# エポック毎にテストデータを作成する関数
def make_epoch_test_data():
    # 検索するデータセットのファイルのtop_pathを指定する
    top_path = os.path.join("CVL_test_data")
    temp_list = [data_filepath for data_filepath in os.walk(top_path)]
    tup = temp_list[0]
    (dirpath, dirnames, filenames) = tup

    image_size = 200
    images = []
    text_names = []
    for filename in filenames:
        image = plt.imread(os.path.join(dirpath, filename))

        # 画像データの名前と拡張子を分離する
        name, ext = os.path.splitext(filename)

        heigh = image.shape[0]
        width = image.shape[1]

        # １人あたり3種類の画像から1枚ずつ切り出し画像を作成する
        x_select_points = width - image_size
        y_select_points = heigh - image_size
        x_select_point = np.random.permutation(x_select_points)
        y_select_point = np.random.permutation(y_select_points)
        y_p = y_select_point[0]
        x_p = x_select_point[0]

        image = image[y_p:y_p+image_size, x_p:x_p+image_size].copy()
        text_name = name[:4]
#        print text_name
        images.append(image)
        text_names.append(text_name)

    x = np.array(images).reshape(-1, 1, image_size, image_size)
    t = np.array(text_names)
    return x, t

if __name__ == '__main__':

    # 超パラメータの定義
    learning_rate = 0.000001  # learning_rate(学習率)を定義する
    max_iteration = 2      # 学習させる回数
    batch_size = 200       # ミニバッチ1つあたりのサンプル数
    dim_hidden_1 = 500         # 隠れ層の次元数を定義する
    dim_hidden_2 = 500
    wscale_1 = 1.0
    wscale_2 = 1.0
    wscale_3 = 1.0
    l_2 = 0.0015

    # 学習させるループ
    for epoch in range(max_iteration):
        print "epoch:", epoch
        x_train_data, t_train_data = make_epoch_train_data()
#        print "x_train_data:", x_train_data
        print "x_train_data.shape:", x_train_data.shape
#        print "t_train_data:", t_train_data
        print "t_train_data.shape:", t_train_data.shape

        x_test_data, t_test_data = make_epoch_test_data()
        print "x_test_data.shape:", x_test_data.shape
        print "t_test_data.shape:", t_test_data.shape

        # mini batchi SGDで重みを更新させるループ

            # 逆伝播

        # 誤差

        # 学習曲線をプロットする

    # 学習済みのモデルをテストセットで誤差と正解率を求める

    # wの可視化
