# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:27:12 2016

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

