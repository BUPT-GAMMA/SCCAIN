import sys

sys.path.append('../')

import numpy as np
import tensorflow as tf
from input import *
from model_all import SCCAIN
from datetime import datetime
from scipy.sparse import coo_matrix


def run_model(data, FLAGS):
    links, FA, GA, FL, GL, FP, FN, GP, GN, FGP, FGN = data


    FC = FLAGS.FC
    GC = FLAGS.GC
    
    global_step = tf.train.get_or_create_global_step()

    md = SCCAIN(links, FN, GN, FP, GP, FGP, FGN, FA, GA, FLAGS, FC, GC, global_step, FLAGS.TP)

    print("start training")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        F = np.random.random([FA.shape[0], FC])
        S = np.random.random([FC, GC])
        G = np.random.random([GA.shape[0], GC])
        XR = sess.run(md.X_rele)

        Ft = []
        Gt = []
        St = []
        FSG = np.dot(np.dot(F, S), G.T)
        print(datetime.now().second)
        for i in range(FLAGS.epochs):
            new_FSG,_, _ = calculate_FSG(F, G, S)
            feed_dict2 = {
                md.FSG2: new_FSG
            }
            for j in range(10):
                _, loss, XR = sess.run(md.opt2, feed_dict=feed_dict2)

            for j in range(10):
                feed_dict = {
                    md.F: F,
                    md.S: S,
                    md.G: G,
                    md.XR: XR
                }
                F, S, G, FSG = sess.run(md.opt1, feed_dict=feed_dict)
            Ft.append(np.argmax(F, axis=1))
            Gt.append(np.argmax(G, axis=1))
    return np.argmax(F, axis=1), np.argmax(G, axis=1), Ft, Gt, FSG


def norm_cluster(matrix):
    idx = np.argmax(matrix, axis=1)
    idx2 = np.arange(matrix.shape[0], dtype=np.int)

    new_matrix = np.array(coo_matrix((np.ones([len(idx)]), (idx2, idx)), shape=matrix.shape).toarray())

    temp = new_matrix.T
    subs = np.sum(np.square(temp), axis=1)
    subs2 = np.sqrt(subs).reshape([-1, 1])

    idxs = np.where(subs > 0)[0]
    temp2 = np.copy(temp)
    temp2[idxs] = temp[idxs] / subs2[idxs]
    return temp2.T


def calculate_FSG(F, G, S):
    new_F = norm_cluster(F)
    new_G = norm_cluster(G)

    FSG = np.dot(np.dot(new_F, S), new_G.T)
    return FSG, new_F, new_G
