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
from chainer import cuda
import chainer.functions as F
from chainer import Variable, FunctionSet
from chainer.optimizers import SGD, Adam


def loss_and_accuracy(model, x_data, t_data, train=False):
    x = Variable(x_data.reshape(-1, 1, 200, 200))
    t = Variable(t_data)

    # 順伝播
    # C25,p2
    h = model.conv_1(x)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # C25,p2
    h = model.conv_2(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # C25,p2
    h = model.conv_3(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # C4
    h = model.conv_4(h)
    h = model.linear_1(h)
    h = F.relu(h)
    a_y = model.linear_2(h)

    loss = F.softmax_cross_entropy(a_y, t)
    accuracy = F.accuracy(a_y, t)

    return loss, cuda.to_cpu(accuracy.data) * 100


def renumber(array):
    """
    input:
        a sorted array-like, e.g. [2, 2, 5, 5, 5, 49, 49, 207, 207, 207]
    output:
        np.ndarray: renumbered array, e.g. [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    """
    is_ndarray = isinstance(array, np.ndarray)
    if is_ndarray:
        shape = array.shape
        dtype = array.dtype
    counts = np.bincount(np.asarray(array).ravel())
    renumbered = [[i] * k for i, k in enumerate(counts[counts.nonzero()])]
    renumbered = np.concatenate(renumbered)
    if is_ndarray:
        return renumbered.astype(dtype).reshape(shape)
    else:
        return renumbered


# エポック毎に訓練データを作成する関数
def make_epoch_train_data():
    # 検索するデータセットのファイルのtop_pathを指定する
    top_path = os.path.join("CVL_ConvNet_data")
    temp_list = [data_filepath for data_filepath in os.walk(top_path)]
    tup = temp_list[0]
    (dirpath, dirnames, filenames) = tup

    image_size = 200
    images = []
    file_numbers = []
    for filename in filenames:

        # 画像データの名前と拡張子を分離する
        name, ext = os.path.splitext(filename)

        # 3種類の訓練画像だけを指定する
        text_type = name[5:6]
        if(int(text_type) > 3):
            continue
        image = plt.imread(os.path.join(dirpath, filename))
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
        images.append(image)
        file_numbers.append(text_name)

    x = np.array(images).reshape(-1, 1, image_size, image_size)
    t = np.array(file_numbers)
    t = t.astype(np.int32)
    t = renumber(t)
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

        image = image[y_p:y_p+image_size, x_p:x_p+image_size].copy()
        text_name = name[:4]
        images.append(image)
        file_numbers.append(text_name)

    x = np.array(images).reshape(-1, 1, image_size, image_size)
    t = np.array(file_numbers)
    t = t.astype(np.int32)
    t = renumber(t)
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

    loss_history = []
    train_accuracy_history = []
    # 学習させるループ
    for epoch in range(max_iteration):
        print "epoch:", epoch
        w_1_grad_norms = []
        w_2_grad_norms = []
        w_3_grad_norms = []
        b_1_grad_norms = []
        b_2_grad_norms = []
        b_3_grad_norms = []
        train_losses = []
        train_accuracies = []

        x_train, t_train = make_epoch_train_data()
        print "x_train.shape:", x_train.shape
        print "t_train:", t_train.shape

        x_test, t_test = make_epoch_test_data()
        print "x_test.shape:", x_test.shape
        print "t_test.shape:", t_test.shape

        num_train = len(x_train)
        num_test = len(x_test)
        classes = np.unique(t_train)  # 定義されたクラスラベル
        num_classes = len(classes)  # クラス数
        dim_features = x_train.shape[-1]  # xの次元

        model = FunctionSet(conv_1=F.Convolution2D(1, 50, 25),
                            conv_2=F.Convolution2D(50, 50, 25),
                            conv_3=F.Convolution2D(50, 100, 25),
                            conv_4=F.Convolution2D(100, 200, 4),
                            linear_1=F.Linear(200, 400, wscale=wscale_1),
                            linear_2=F.Linear(400, num_classes,
                                              wscale=wscale_2)).to_gpu()
        num_train_batches = num_train / batch_size  # ミニバッチの個数

        # mini batchi SGDで重みを更新させるループ
        time_start = time.time()
        perm_train = np.random.permutation(num_train)

        for batch_indexes in np.array_split(perm_train[:100], num_train_batches):
            x_batch = cuda.to_gpu(x_train[batch_indexes])
            t_batch = cuda.to_gpu(t_train[batch_indexes])

            batch_loss, batch_accuracy = loss_and_accuracy(model,
                                                           x_batch, t_batch,
                                                           train=True)

            # 逆伝播

        # 誤差

        # 学習曲線をプロットする

    # 学習済みのモデルをテストセットで誤差と正解率を求める

    # wの可視化
