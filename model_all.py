import sys

import numpy as np
import tensorflow as tf
from input import *


class SCCAIN:
    def __init__(self, links, FN, GN, FP, GP, FGP, FGN, FA, GA, FLAGS, FC, GC, global_step, types=1):
        NF, NFA = FA.shape
        NG, NGA = GA.shape
        ND = FLAGS.ND

        D = 5

        W = tf.get_variable(name="W", dtype=tf.float32, shape=[links.shape[0], 1, 1])

        if types == 1:
            links = np.array([m.toarray() for m in links[0]])
        XL = tf.constant(links, dtype=tf.float32, name="XL")
        W2 = tf.nn.relu(W) + 0.01
        W2 = W2 / tf.reduce_sum(W2)


        FAtt = tf.constant(FA, dtype=tf.float32, shape=[NF, NFA])
        GAtt = tf.constant(GA, dtype=tf.float32, shape=[NG, NGA])
        b1 = tf.get_variable(name="b1", dtype=tf.float32, shape=(1,), initializer=tf.zeros_initializer())
        Phi1 = tf.get_variable(dtype=tf.float32, shape=[NFA, NGA],
                               initializer=tf.glorot_uniform_initializer(seed=FLAGS.seed),
                               name="project_matrix1")

        alpha = FLAGS.times * FLAGS.kd

        self.XA = tf.nn.relu(tf.tanh(tf.matmul(tf.matmul(FAtt, Phi1), GAtt, transpose_b=True) + b1)) + 0.0001

        self.XL = tf.reduce_sum(tf.multiply(XL, W2), axis=0)
        self.X_rele = (1 - alpha) * self.XL + alpha * self.XA

        self.F = tf.placeholder(tf.float32, shape=[NF, FC], name="F")
        self.S = tf.placeholder(tf.float32, shape=[FC, GC], name="S")
        self.G = tf.placeholder(tf.float32, shape=[NG, GC], name="G")
        self.XR = tf.placeholder(tf.float32, shape=[NF, NG], name="XR")

        # ---------------------------------------------
        N1_F = create_sparseTensor(FN, NF)
        P1_F = create_sparseTensor(FP, NF)

        N1_G = create_sparseTensor(GN, NG)
        P1_G = create_sparseTensor(GP, NG)
        # ---------------------------------------------

        P_F = tf.sparse_tensor_dense_matmul(tf.sparse_reorder(P1_F), self.F)
        # N_F = tf.sparse_tensor_dense_matmul(tf.sparse_reorder(N1_F), self.F)
        P_G = tf.sparse_tensor_dense_matmul(tf.sparse_reorder(P1_G), self.G)
        # N_G = tf.sparse_tensor_dense_matmul(tf.sparse_reorder(N1_G), self.G)
        # P_F = 0
        # P_G = 0

        # -----------------------NMF optimizer-------------------------------

        up_F = tf.matmul(tf.matmul(self.XR, self.G), self.S, transpose_b=True) + P_F
        down_F = tf.matmul(self.F, tf.matmul(self.F, (up_F), transpose_a=True))


        self.new_F = tf.multiply(tf.sqrt(tf.divide(up_F, down_F)), self.F)

        up_G = tf.matmul(self.XR, tf.matmul(self.new_F, self.S), transpose_a=True) + P_G
        down_G = tf.matmul(self.G, tf.matmul(self.G, (up_G), transpose_a=True))

        self.new_G = tf.multiply(tf.divide(up_G, down_G), self.G)
        # self.new_G = tf.Print(self.new_G, ["G", tf.reduce_sum(self.new_G)])

        p1_S = tf.matmul(tf.matmul(self.new_F, self.XR, transpose_a=True), self.new_G)
        p2_S = tf.matmul(self.new_F, self.new_F, transpose_a=True)
        p3_S = tf.matmul(self.new_G, self.new_G, transpose_a=True)

        p4_S = tf.matmul(tf.matmul(p2_S, self.S), p3_S)
        p5_S = tf.sqrt(tf.divide(p1_S, p4_S))
        self.new_S = tf.multiply(p5_S, self.S)

        x1 = self.new_F
        x2 = self.new_G

        self.FSG = tf.matmul(tf.matmul(x1, self.new_S), x2, transpose_b=True)

        self.opt1 = [self.new_F, self.new_S, self.new_G, self.FSG]

        self.FSG2 = tf.placeholder(dtype=tf.float32, shape=[NF, NG], name="FSG2")

        self.loss3 = tf.reduce_mean(tf.square(self.XR - self.FSG))

        # -------------------relevance optimization

        loss1 = tf.reduce_mean(tf.square(self.X_rele - self.FSG2))
        loss2 = tf.reduce_mean(tf.multiply(Phi1, Phi1)) + tf.reduce_mean(
            tf.multiply(b1, b1)) + tf.reduce_mean(tf.multiply(W, W))

        loss3 = self.loss_must_cannot(FGP, FGN, self.X_rele, NF, NG)


        self.loss = FLAGS.lambdas * loss2 + loss3 + loss1


        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss,
                                                                                       global_step=global_step)
        self.opt2 = [optimizer, self.loss, self.X_rele]

    def loss_must_cannot(self, Pos, Neg, X, a, b):
        pos_matrix = self.create_sparse_dense_tensor(Pos, a, b)
        neg_matrix = self.create_sparse_dense_tensor(Neg, a, b)
        x1 = tf.reduce_mean(tf.multiply(pos_matrix, X))
        x2 = tf.reduce_mean(tf.multiply(neg_matrix, X))

        return x2 - x1

    def create_sparse_dense_tensor(self, indices, a, b):
        spm = tf.SparseTensor(indices=indices, values=tf.ones(len(indices), dtype=tf.float32), dense_shape=[a, b])
        spm = tf.sparse_reorder(spm)
        m = tf.sparse_to_dense(sparse_indices=spm.indices, sparse_values=spm.values, output_shape=spm.dense_shape)
        return m
