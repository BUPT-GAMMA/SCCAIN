

# input datas

import sys
sys.path.append('../')

from tools.basic_tools import *
import tensorflow as tf
import numpy as np


def load_data(path, dataname, supername, rate):
    datas = load_mat_data(path + dataname)
    sinfos = load_mat_data(path + "rate_{}_".format(rate) + supername)

    FA = datas["FA"]
    GA = datas["GA"]
    links = datas["links"]
    FL = datas["FL"]
    GL = datas["GL"]

    FP = sinfos["FP"]
    FN = sinfos["FN"]
    GP = sinfos["GP"]
    GN = sinfos["GN"]
    FGP = sinfos["FGP"]
    FGN = sinfos["FGN"]

    return links, FA, GA, FL[0], GL[0], FP, FN, GP, GN, FGP, FGN

def load_data2(path, dataname, supername, rate):
    datas = load_mat_data(path + dataname)
    sinfos = load_mat_data(path + "rate_{}_".format(rate) + supername)

    FA = datas["FA"]
    GA = datas["GA"]
    links = datas["links"]
    FL = datas["FL"]
    GL = datas["GL"]

    FP = sinfos["FP"]
    FN = sinfos["FN"]
    GP = sinfos["GP"]
    GN = sinfos["GN"]
    FGP = sinfos["FGP"]
    FGN = sinfos["FGN"]

    return links, FA, GA, FL, GL, FP, FN, GP, GN, FGP, FGN


def arr2sparse(arr_tensor):
    arr_idx = tf.where(tf.not_equal(arr_tensor, 0))
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_tensor.get_shape())
    return arr_sparse


def create_sparseTensor(lists, M):
    arrs = np.array(lists, np.int32)
    data = [1.0] * (len(lists) * 2)

    x = arrs[:, 0].tolist()
    y = arrs[:, 1].tolist()

    rows = x + y
    cols = y + x
    indices = np.array([rows, cols]).T

    return tf.SparseTensor(indices, data, dense_shape=[M, M])
