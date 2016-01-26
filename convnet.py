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
import chainer.optimizers
from chainer import cuda
from chainer.optimizers import SGD, Adam


def loss_and_accuracy(model, x_data, t_data, train=False):
    x = Variable(x_data.reshape(-1, 1, 200, 200))
    t = Variable(t_data)

    # 順伝播
    # 1.C3,p2
    h = model.conv_1(x)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 2.C4,p2
    h = model.conv_2(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 3.C5,p2
    h = model.conv_3(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 4.C3,p2
    h = model.conv_4(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 5.C3,p2
    h = model.conv_5(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 6.C3,p2
    h = model.conv_6(h)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
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
    learning_rate = 0.01  # learning_rate(学習率)を定義する
    max_iteration = 20      # 学習させる回数
    batch_size = 50       # ミニバッチ1つあたりのサンプル数
    dim_hidden_1 = 500         # 隠れ層の次元数を定義する
    dim_hidden_2 = 500
    wscale_1 = 1.0
    wscale_2 = 1.0
    wscale_3 = 1.0
    l_2 = 0.0015
    train_accuracy_best = 0
    train_loss_best = 10

    # 訓練データに必要な定義をする
    x_train, t_train = make_epoch_train_data()
    num_train = len(x_train)
    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元
    num_train_batches = num_train / batch_size  # ミニバッチの個数

    # テストデータに必要な定義をする
    x_test, t_test = make_epoch_test_data()
    num_test = len(x_test)
    dim_features = x_train.shape[-1]  # xの次元
    num_test_batches = num_test / batch_size  # ミニバッチの個数

    # モデルの定義をする
    model = FunctionSet(conv_1=F.Convolution2D(1, 50, 3),
                        conv_2=F.Convolution2D(50, 50, 4),
                        conv_3=F.Convolution2D(50, 100, 5),
                        conv_4=F.Convolution2D(100, 200, 3),
                        conv_5=F.Convolution2D(200, 100, 3),
                        conv_6=F.Convolution2D(100, 200, 3),
                        linear_1=F.Linear(200, 400, wscale=wscale_1),
                        linear_2=F.Linear(400, num_classes,
                                          wscale=wscale_2)).to_gpu()

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
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
        test_losses = []
        test_accuracies = []

        x_train, t_train = make_epoch_train_data()
        print "x_train.shape:", x_train.shape
        print "t_train:", t_train.shape

        x_test, t_test = make_epoch_test_data()
        print "x_test.shape:", x_test.shape
        print "t_test.shape:", t_test.shape

        optimizer = chainer.optimizers.Adam(learning_rate)
        optimizer.setup(model)

        # mini batchi SGDで重みを更新させるループ
        time_start = time.time()
        perm_train = np.random.permutation(num_train)
        sort_train = np.sort(perm_train)
        for batch_indexes in np.array_split(sort_train[:100],
                                            num_train_batches):
            x_batch = cuda.to_gpu(x_train[batch_indexes])
            t_batch = cuda.to_gpu(t_train[batch_indexes])

            batch_loss, batch_accuracy = loss_and_accuracy(model,
                                                           x_batch, t_batch,
                                                           train=True)

            # 逆伝播
            optimizer.zero_grads()
            batch_loss.backward()
            optimizer.update()

            w_1_grad_norm = np.linalg.norm(model.linear_1.W.grad.get())
            w_1_grad_norms.append(w_1_grad_norm)
            w_2_grad_norm = np.linalg.norm(model.linear_2.W.grad.get())
            w_2_grad_norms.append(w_2_grad_norm)

        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print "time_elapsed:", time_elapsed

        # 誤差
        # 訓練データセットの交差エントロピー誤差と正解率を表示する
        for batch_indexes in np.array_split(sort_train[:100],
                                            num_train_batches):
            x_batch_train = cuda.to_gpu(x_train[batch_indexes])
            t_batch_train = cuda.to_gpu(t_train[batch_indexes])

            train_loss, train_accuracy = loss_and_accuracy(model,
                                                           x_batch_train,
                                                           t_batch_train)
            train_losses.append(train_loss.data.get())
            train_accuracies.append(train_accuracy)
        average_train_loss = np.array(train_losses).mean()
        average_train_accuracy = np.array(train_accuracies).mean()
        print "[train] Loss:", average_train_loss
        print "[train] Accuracy:", average_train_accuracy
        loss_history.append(average_train_loss)
        train_accuracy_history.append(average_train_accuracy)

        # 学習曲線をプロットする
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(loss_history)
        plt.legend(["train"], loc="best")
        plt.ylim([0.0, 0.4])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(train_accuracy_history)
        plt.legend(["train"], loc="best")
        plt.ylim([0, 100])
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.draw()

        # 訓練データの誤差が良ければwの最善値を保存する
        if train_loss.data <= train_loss_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            train_loss_best = train_loss.data
            train_accuracy_best = train_accuracy
            print "epoch_best:", epoch_best
            print "train_loss_best:", train_loss_best
            print "train_accuracy_best:", train_accuracy_best
            print

    # 学習済みのモデルをテストセットで誤差と正解率を求める
    perm_test = np.random.permutation(num_test)
    sort_test = np.sort(perm_test)
    for batch_indexes in np.array_split(sort_test[:30],
                                        num_test_batches):
        x_batch_test = cuda.to_gpu(x_test[batch_indexes])
        t_batch_test = cuda.to_gpu(t_test[batch_indexes])

        test_loss, test_accuracy = loss_and_accuracy(model_best,
                                                      x_batch_test,
                                                      t_batch_test)
        test_losses.append(test_loss.data.get())
        test_accuracies.append(test_accuracy)
    average_test_loss = np.array(test_losses).mean()
    average_test_accuracy = np.array(test_accuracies).mean()

    print "[test]  Accuracy:", test_accuracy
    print "[train] Loss:", train_loss.data
    print "Best epoch :", epoch_best
    print "Finish epoch:", epoch
    print "Batch size:", batch_size
    print "Learning rate:", learning_rate
    print "dim_hidden_1:", dim_hidden_1
    print "dim_hidden_2:", dim_hidden_2
    print "wscale_1:", wscale_1
    print "wscale_2:", wscale_2
    print "wscale_3:", wscale_3
    print "l_2:", l_2

    print

    # wの可視化
