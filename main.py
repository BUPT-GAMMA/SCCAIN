import sys

import numpy as np
import tensorflow as tf

from input import *
import main_all as model0
from basic_tools import *
from measure import *

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("data_path", './data/dblp/SCCAIN/', "data path")
flags.DEFINE_string("suffix", '', 'suffix')
flags.DEFINE_float("super_rate", 0.1, "super_rate")
flags.DEFINE_integer("model", 2, "model")
flags.DEFINE_integer("FC", 4, "F clusters")
flags.DEFINE_integer("GC", 4, "G clusters")

flags.DEFINE_string("output", "../output_dblp/", "output")
flags.DEFINE_integer("train", 1, "train/test")
flags.DEFINE_integer("epochs", 10, "epochs")
flags.DEFINE_float("kd", 0.05, "kd")
flags.DEFINE_integer("seed", 123, "seed")
flags.DEFINE_integer("sd", 123, "sd")
flags.DEFINE_float("learning_rate", 0.1, "learning rate")
flags.DEFINE_integer("times", 0, "times of alpha")
flags.DEFINE_float("lambdas", 0.1, "lambda")
flags.DEFINE_integer("ND", 4, "ND")
flags.DEFINE_integer("TP", 0, "TP")

tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)


def main(argv=None):
    store_path = FLAGS.output + "sccain_model_{}_rate_{}_alpha_{}{}_sd{}/".format(FLAGS.model, FLAGS.super_rate, FLAGS.times,
                                                                             FLAGS.suffix, FLAGS.sd)
    create_directory(store_path)
    filename = store_path + "result.pkl"
    filename_m = store_path + "m.pkl"
    data = load_data2(FLAGS.data_path, "data.mat", "seed_{}superinfo.mat".format(FLAGS.sd), FLAGS.super_rate)
    opts = model0

    FL, GL = data[3], data[4]
    F, G, Fts, GTs, FSG = opts.run_model(data, FLAGS)
    store_pkl(filename, [F, G, Fts, GTs, FL, GL, FSG])
    infos = measure_result(FL, GL, F, G)
    store_pkl(filename_m, infos)
    print(infos)


if __name__ == "__main__":
    tf.app.run()
