# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 19:04:27 2016

@author: matsumi
"""

import load_mnist
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time
import copy
import chainer.functions as F
from chainer import Variable, FunctionSet
import chainer.optimizers
from chainer import cuda
from chainer.cuda import cupy
import random

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
    x_train, t_train, x_test, t_test = load_mnist.load_mnist()
    t_train = t_train.astype(np.int32)
    t_test = t_test.astype(np.int32)
    plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    plt.show()

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape

    # 60000ある訓練データセットを50000と10000の評価のデータセットに分割する
    x_train, x_valid, t_train, t_valid = train_test_split(
        x_train, t_train, test_size=0.1, random_state=100)

    print "x_train.shape:", x_train.shape
    print "t_train.shape:", t_train.shape
    print "x_valid.shape:", x_valid.shape
    print "t_valid.shape:", t_valid.shape
    print "x_test.shape:", x_test.shape
    print "t_test.shape:", t_test.shape

#    num_train = len(x_train)
#    num_valid = len(x_valid)
#    num_test = len(x_test)
    # 学習させるサンプル数を減らす
    num_train = 6
    num_valid = 6
    num_test = 6

    classes = np.unique(t_train)  # 定義されたクラスラベル
    classes = classes[:2]
    num_classes = len(classes)  # クラス数
    dim_features = x_train.shape[-1]  # xの次元

    # 超パラメータの定義
    learning_rate = 0.000001  # learning_rate(学習率)を定義する
    max_iteration = 100      # 学習させる回数
    batch_size = 3       # ミニバッチ1つあたりのサンプル数
    dim_hidden_1 = 500         # 隠れ層の次元数を定義する
    dim_hidden_2 = 500
    wscale_1 = 1.0
    wscale_2 = 1.0
    wscale_3 = 1.0
    l_2 = 0.0015

    model = FunctionSet(conv_11=F.Convolution2D(1, 50, 5),
                        conv_12=F.Convolution2D(50, 50, 1),
#                        conv_13=F.Convolution2D(50, 50, 1),
                        conv_2=F.Convolution2D(50, 100, 5),
                        conv_3=F.Convolution2D(100, 200, 4),
                        linear_1=F.Linear(200, 400, wscale=wscale_1),
                        linear_2=F.Linear(400, num_classes,
                                          wscale=wscale_2)).to_gpu()

    optimizer = chainer.optimizers.Adam(learning_rate)
    optimizer.setup(model)

    loss_history = []
    train_accuracy_history = []
    loss_valid_history = []
    valid_accuracy_history = []

    valid_accuracy_best = 0
    valid_loss_best = 10
    num_train_batches = num_train / batch_size  # ミニバッチの個数
    num_valid_batches = num_valid / batch_size
    num_mini_train = []
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

        # mini batchi SGDで重みを更新させるループ
        time_start = time.time()
        x_train, t_train, x_test, t_test = load_mnist.load_mnist()
        t_train = t_train.astype(np.int32)
        t_test = t_test.astype(np.int32)
        x_train = x_train[:12600]
        t_train = t_train[:12600]
        for i in range(6):
            num_mini_train.append(random.choice(t_train))
#        perm_train = np.random.permutation(num_train)
#        sort_train = np.sort(perm_train)

        print num_mini_train
        for batch_indexes in np.array_split(num_mini_train, num_train_batches):
            x_batch = cuda.to_gpu(x_train[batch_indexes])
            t_batch = cuda.to_gpu(t_train[batch_indexes])

            batch_loss, batch_accuracy = loss_and_accuracy(model,
                                                           x_batch, t_batch,
                                                           train=True)
            # 逆伝播
            optimizer.zero_grads()
            batch_loss.backward()
#            optimizer.weight_decay(l_2)
            optimizer.update()

            w_1_grad_norm = np.linalg.norm(model.linear_1.W.grad.get())
            w_1_grad_norms.append(w_1_grad_norm)
            w_2_grad_norm = np.linalg.norm(model.linear_2.W.grad.get())
            w_2_grad_norms.append(w_2_grad_norm)
#            w_3_grad_norm = np.linalg.norm(model.linear_3.W.grad.get())
#            w_3_grad_norms.append(w_3_grad_norm)
#
#            b_1_grad_norm = np.linalg.norm(model.linear_1.b.grad.get())
#            b_1_grad_norms.append(b_1_grad_norm)
#            b_2_grad_norm = np.linalg.norm(model.linear_2.b.grad.get())
#            b_2_grad_norms.append(b_2_grad_norm)
#            b_3_grad_norm = np.linalg.norm(model.linear_3.b.grad.get())
#            b_3_grad_norms.append(b_3_grad_norm)

        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print "time_elapsed:", time_elapsed

        # 誤差
        # E(K×K)を出す0.5×(y-t)×(y-t).T次元数は，{0.5×(1×K)(K×1)}
        # E = sum(t×log(y)(1×K))
        # 訓練データセットの交差エントロピー誤差と正解率を表示する
        for batch_indexes in np.array_split(perm_train, num_train_batches):
            x_batch_train = cuda.to_gpu(x_train[batch_indexes])
            t_batch_train = cuda.to_gpu(t_train[batch_indexes])

            train_loss, train_accuracy = loss_and_accuracy(model,
                                                           x_batch_train,
                                                           t_batch_train)
            train_losses.append(train_loss.data)
            train_accuracies.append(train_accuracy)
#            train_mean = cupy.mean(train_loss.data)

#        train_mean_accuracies = cupy.mean(train_accuracies, dtype=cupy.float32)
        print " |W_1|", np.linalg.norm(cuda.to_cpu(model.linear_1.W.data.get()))
        print "w_1_grad_norm", w_1_grad_norm
        print " |W_2|", np.linalg.norm(cuda.to_cpu(model.linear_2.W.data.get()))
        print "w_2_grad_norm", w_2_grad_norm
        print "[train] Loss:", train_loss.data
        print "[train] Accuracy:", train_accuracy
        loss_history.append(train_loss.data.get())
        train_accuracy_history.append(train_accuracy)

        # 検証データセットの交差エントロピー誤差と正解率を表示する
        perm_valid = np.random.permutation(num_valid)

        for batch_indexes in np.array_split(perm_valid, num_valid_batches):
            x_batch_valid = cuda.to_gpu(x_valid[batch_indexes])
            t_batch_valid = cuda.to_gpu(t_valid[batch_indexes])

            valid_loss, valid_accuracy = loss_and_accuracy(model,
                                                           x_batch_valid,
                                                           t_batch_valid)
        print "[valid] Loss:", valid_loss.data
        print "[valid] Accuracy:", valid_accuracy
        print
        loss_valid_history.append(valid_loss.data.get())
        valid_accuracy_history.append(valid_accuracy)

        print "w_1_grad_mean:", np.mean(w_1_grad_norms, dtype=np.float32)
        print "w_2_grad_mean:", np.mean(w_2_grad_norms, dtype=np.float32)
#        print "w_3_grad_mean:", np.mean(w_3_grad_norms, dtype=np.float32)
#        print "b_1_grad_mean:", np.mean(b_1_grad_norms, dtype=np.float32)
#        print "b_2_grad_mean:", np.mean(b_2_grad_norms, dtype=np.float32)
#        print "b_3_grad_mean:", np.mean(b_3_grad_norms, dtype=np.float32)
#        print "|w_1|:", np.linalg.norm(model.linear_1.W.data.get())
#        print "|w_2|:", np.linalg.norm(model.linear_2.W.data.get())
#        print "|w_3|:", np.linalg.norm(model.linear_3.W.data.get())
#        print "|b_1|:", np.linalg.norm(model.linear_1.b.data.get())
#        print "|b_2|:", np.linalg.norm(model.linear_2.b.data.get())
#        print "|b_3|:", np.linalg.norm(model.linear_3.b.data.get())

        # 学習曲線をプロットする
        # plot learning curves
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(loss_history)
        plt.plot(loss_valid_history)
        plt.legend(["train", "valid"], loc="best")
        plt.ylim([0.0, 0.4])
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(train_accuracy_history)
        plt.plot(valid_accuracy_history)
        plt.legend(["train", "valid"], loc="best")
        plt.ylim([91, 100])
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.draw()

        # 検証データの誤差が良ければwの最善値を保存する
        if valid_loss.data <= valid_loss_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            valid_loss_best = valid_loss.data
            valid_accuracy_best = valid_accuracy
            print "epoch_best:", epoch_best
            print "valid_loss_best:", valid_loss_best
            print "valid_accuracy_best:", valid_accuracy_best
            print
    # 学習済みのモデルをテストセットで誤差と正解率を求める

    test_error, test_accuracy = loss_and_accuracy(model_best,
                                                  cuda.to_gpu(x_test[0]),
                                                  cuda.to_gpu(t_test[0]))

    print "[test]  Accuracy:", test_accuracy
    print "[valid] Accuracy (best)  :", valid_accuracy_best
    print "[valid] Loss (best):", valid_loss_best
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
    print "|w_1_best|:", np.linalg.norm(model.linear_1.W.data)
    # print "w_1_best:", model.linear_1.W.data
    print "|w_2_best|:", np.linalg.norm(model.linear_2.W.data)
    # print "w_2_best:", model.linear_2.W.data
    print "|w_3_best|:", np.linalg.norm(model.linear_3.W.data)
    # print "w_3_best:", model.linear_3.W.data
    w_best = np.dot(model.linear_2.W.data, model.linear_1.W.data)
    w_best = np.dot(model.linear_3.W.data, w_best)
    fig, axes = plt.subplots(2, 5,  figsize=(10, 4))
    for w_k, ax in zip(w_best, axes.ravel()):
        w_true = w_k[0:784]  # w_trueとは結果をプロットするために定義したものである
        ax.matshow(w_true.reshape(28, 28), cmap=plt.cm.gray)
    plt.show()
