
import os
import scipy.io as sio
import pickle as pkl
import math


def create_directory(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print("Complete create directory:\t{}".format(pathname))
    else:
        print("The directory:\t {} has been created".format(pathname))
    return 0


def load_mat_data(filename):
    return sio.loadmat(filename)


def store_pkl(filename, data):
    print(filename)
    return pkl.dump(data, open(filename, "wb"))


def load_pkl(filename):
    return pkl.load(open(filename, "rb"))


def store_data_mat(filename, path, formated_data):
    create_directory(path)
    sio.savemat(path + filename, formated_data)
    print("stored {}".format(path + filename))
    return "OK"


def store_data_string(filename, path, data):
    create_directory(path)
    with open(path + filename, "w") as f:
        for d in data:
            f.write(d + "\n")


def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s
