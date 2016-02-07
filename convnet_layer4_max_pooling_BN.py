# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 22:36:01 2016

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
from chainer import serializers

def loss_and_accuracy(model, x_data, t_data, train=False):
    x = Variable(x_data.reshape(-1, 1, 200, 200))
    t = Variable(t_data)

    # 順伝播
    # 1.C3,p2
    h = model.conv_1(x)
    h = model.bn1(h, test=not train)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 2.C4,p2
    h = model.conv_2(h)
    h = model.bn2(h, test=not train)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 3.C5,p2
    h = model.conv_3(h)
    h = model.bn3(h, test=not train)
    h = F.max_pooling_2d(h, 2)
    h = F.relu(h)
    # 4.C3
    h = model.conv_4(h)
    h = model.bn4(h, test=not train)
    h = F.relu(h)
    # 4-5.C1
    h = model.conv_4_5(h)
    h = model.bn4_5(h, test=not train)
    h = F.relu(h)
    # 4-5_2.C1,p2
    h = model.conv_4_5_2(h)
    h = F.max_pooling_2d(h, 20)
    h = F.relu(h)
    # linear_1
    h = model.linear_1(h)
    h = F.relu(h)
    h = F.dropout(h, ratio=0.9, train=train)
    # linear_2
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
def make_epoch_train_data(num_classes=308):
    # 削除したいファイル名を指定する
    exclusion_filenames = ["0431-1-cropped.png", "0431-2-cropped.png",
                           "0431-3-cropped.png", "0431-4-cropped.png",
                           "0612-1-cropped.png", "0612-2-cropped.png",
                           "0612-3-cropped.png", "0612-4-cropped.png"]
    # 検索するデータセットのファイルのtop_pathを指定する
    top_path = os.path.join("CVL_ConvNet_data")
    temp_list = [data_filepath for data_filepath in os.walk(top_path)]
    tup = temp_list[0]
    (dirpath, dirnames, filenames) = tup

    lower_text = 1250
    image_size = 200
    images = []
    file_numbers = []
    # 削除したいファイルを削除する
    filenames = sorted(list(set(filenames) - set(exclusion_filenames)))
    num_files = num_classes * 4
    filenames = filenames[:num_files]
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

        # 決められた文字の量が切り出し画像に含まれるようにする
        while True:
            x_select_point = np.random.permutation(x_select_points)
            y_select_point = np.random.permutation(y_select_points)
            y_p = y_select_point[0]
            x_p = x_select_point[0]
            cropped_image = image[y_p:y_p+image_size,
                                  x_p:x_p+image_size].copy()
            if np.count_nonzero(cropped_image) > lower_text:
                break
        text_name = name[:4]
        images.append(cropped_image)
        file_numbers.append(text_name)

    x = np.array(images).reshape(-1, 1, image_size, image_size)
    t = np.array(file_numbers)
    t = t.astype(np.int32)
    t = renumber(t)
    return x, t


# エポック毎にテストデータを作成する関数
def make_epoch_test_data(num_classes=308):
    # 削除したいファイル名を指定する
    exclusion_filenames = ["0431-1-cropped.png", "0431-2-cropped.png",
                           "0431-3-cropped.png", "0431-4-cropped.png",
                           "0612-1-cropped.png", "0612-2-cropped.png",
                           "0612-3-cropped.png", "0612-4-cropped.png"]
    # 検索するデータセットのファイルのtop_pathを指定する
    top_path = os.path.join("CVL_test_data")
    temp_list = [data_filepath for data_filepath in os.walk(top_path)]
    tup = temp_list[0]
    (dirpath, dirnames, filenames) = tup

    lower_text = 1250
    image_size = 200
    images = []
    file_numbers = []
    # 削除したいファイルを削除する
    filenames = sorted(list(set(filenames) - set(exclusion_filenames)))
    filenames = filenames[:num_classes]
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

        # 決められた文字の量が切り出し画像に含まれるようにする
        while True:
            x_select_point = np.random.permutation(x_select_points)
            y_select_point = np.random.permutation(y_select_points)
            y_p = y_select_point[0]
            x_p = x_select_point[0]
            cropped_image = image[y_p:y_p+image_size,
                                  x_p:x_p+image_size].copy()
            if np.count_nonzero(cropped_image) > lower_text:
                break
        text_name = name[:4]
        images.append(cropped_image)
        file_numbers.append(text_name)

    x = np.array(images).reshape(-1, 1, image_size, image_size)
    t = np.array(file_numbers)
    t = t.astype(np.int32)
    t = renumber(t)
    return x, t


if __name__ == '__main__':

    # 超パラメータの定義
    learning_rate = 0.0001  # learning_rate(学習率)を定義する
    max_iteration = 2000      # 学習させる回数
    batch_size = 10       # ミニバッチ1つあたりのサンプル数
    wscale_1 = 1.0
    wscale_2 = 1.0
    l_2 = 0.0015
    test_accuracy_best = 0
    test_loss_best = 10
    num_classes = 308

    # 訓練データに必要な定義をする
    x_train, t_train = make_epoch_train_data(num_classes)
    num_train = len(x_train)
    classes = np.unique(t_train)  # 定義されたクラスラベル
    num_classes = len(classes)  # クラス数
    num_train_batches = 1 + (num_train / batch_size)  # ミニバッチの個数

    # テストデータに必要な定義をする
    x_test, t_test = make_epoch_test_data(num_classes)
    num_test = len(x_test)    # 学習させるサンプル数を減らしてみる
    num_test_batches = 1 + (num_test / batch_size)  # ミニバッチの個数

    # モデルの定義をする
    model = FunctionSet(conv_1=F.Convolution2D(1, 50, 3),
                        bn1=F.BatchNormalization(50),
                        conv_2=F.Convolution2D(50, 100, 4),
                        bn2=F.BatchNormalization(100),
                        conv_3=F.Convolution2D(100, 100, 5),
                        bn3=F.BatchNormalization(100),
                        conv_4=F.Convolution2D(100, 200, 3),
                        bn4=F.BatchNormalization(200),
                        conv_4_5=F.Convolution2D(200, 200, 1),
                        bn4_5=F.BatchNormalization(200),
                        conv_4_5_2=F.Convolution2D(200, 200, 1),
                        linear_1=F.Linear(200, 400, wscale=wscale_1),
                        linear_2=F.Linear(400, num_classes,
                                          wscale=wscale_2)).to_gpu()

    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    loss_test_history = []
    final_test_losses = []
    final_test_accuracies = []
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

        # 訓練データとテストデータを呼ぶ
        x_train, t_train = make_epoch_train_data(num_classes)
        x_test, t_test = make_epoch_test_data(num_classes)

        optimizer = chainer.optimizers.Adam(learning_rate)
        optimizer.setup(model)

        # mini batchi SGDで重みを更新させるループ
        time_start = time.time()
        perm_train = np.random.permutation(num_train)
        for batch_indexes in np.array_split(perm_train,
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
        for batch_indexes in np.array_split(np.arange(num_train),
                                            num_train_batches):
            x_batch_train = cuda.to_gpu(x_train[batch_indexes])
            t_batch_train = cuda.to_gpu(t_train[batch_indexes])

            train_loss, train_accuracy = loss_and_accuracy(model,
                                                           x_batch_train,
                                                           t_batch_train)
            train_losses.append(train_loss.data.get())
            train_accuracies.append(train_accuracy)

        # w_1,w_2のノルムを表示する
        print " |W_1|", np.linalg.norm(model.linear_1.W.data.get())
        print "w_1_grad_norm", w_1_grad_norm
        print " |W_2|", np.linalg.norm(model.linear_2.W.data.get())
        print "w_2_grad_norm", w_2_grad_norm
        print " |W|", [np.linalg.norm(w.get()) for w in model.parameters]
        print " W_grad", [np.linalg.norm(w.get()) for w in model.gradients]

        average_train_loss = np.array(train_losses).mean()
        average_train_accuracy = np.array(train_accuracies).mean()
        print "[train] Loss:", average_train_loss
        print "[train] Accuracy:", average_train_accuracy
        loss_history.append(average_train_loss)
        train_accuracy_history.append(average_train_accuracy)

        # テストデータセットの交差エントロピー誤差と正解率を表示する
        for batch_indexes in np.array_split(np.arange(num_test),
                                            num_test_batches):
            x_batch_test = cuda.to_gpu(x_test[batch_indexes])
            t_batch_test = cuda.to_gpu(t_test[batch_indexes])

            test_loss, test_accuracy = loss_and_accuracy(model,
                                                         x_batch_test,
                                                         t_batch_test)
            test_losses.append(test_loss.data.get())
            test_accuracies.append(test_accuracy)
        average_test_loss = np.array(test_losses).mean()
        average_test_accuracy = np.array(test_accuracies).mean()
        loss_test_history.append(average_test_loss)
        test_accuracy_history.append(average_test_accuracy)
        print "[test] Loss:", average_test_loss
        print "[test]  Accuracy:", average_test_accuracy

        # 学習曲線をプロットする
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(loss_test_history, 'g', label="test")
        plt.plot(loss_history, 'b', label="train")
        plt.legend(loc="best")
#        plt.ylim([0.0, 1])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(test_accuracy_history, 'g', label="test")
        plt.plot(train_accuracy_history, 'b', label="train")
        plt.legend(loc="best")
        plt.ylim([0, 100])
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.draw()

        # テストデータの誤差が良ければwの最善値を保存する
        if test_loss.data <= test_loss_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            test_loss_best = test_loss.data
            test_accuracy_best = test_accuracy
            serializers.save_npz("convnet_layer4_max_pooling_BN.npz", model)
            print "epoch_best:", epoch_best
            print "test_loss_best:", test_loss_best
            print "test_accuracy_best:", test_accuracy_best
            print

    # 学習済みのモデルをテストセットで誤差と正解率を求める
    for batch_indexes in np.array_split(np.arange(num_test),
                                        num_test_batches):
        f_x_batch_test = cuda.to_gpu(x_test[batch_indexes])
        f_t_batch_test = cuda.to_gpu(t_test[batch_indexes])

        final_test_loss, final_test_accuracy = loss_and_accuracy(model_best,
                                                                 f_x_batch_test,
                                                                 f_t_batch_test)
        final_test_losses.append(final_test_loss.data.get())
        final_test_accuracies.append(final_test_accuracy)
    average_final_test_loss = np.array(final_test_losses).mean()
    average_final_test_accuracy = np.array(final_test_accuracies).mean()

    print "[final_test]  Accuracy:", final_test_accuracy
    print "[train] Loss:", train_loss.data
    print "test_loss_best:", test_loss_best
    print "Best epoch :", epoch_best
    print "Finish epoch:", epoch
    print "Batch size:", batch_size
    print "Learning rate:", learning_rate
    print "wscale_1:", wscale_1
    print "wscale_2:", wscale_2
    print "l_2:", l_2

    print
