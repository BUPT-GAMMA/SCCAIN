import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import os
from tools.basic_tools import *


def measure(FC, GC, SL, TL):
    nmi_s = NMI(SL, FC)
    nmi_t = NMI(TL, GC)
    ari_s = ARI(SL, FC)
    ari_t = ARI(TL, GC)

    # print(len(set(FC)), len(set(GC)))
    pri_s = purity(FC, SL)
    pri_t = purity(GC, TL)
    ps, rs, fs = PRF1(FC, SL)
    pt, rt, ft = PRF1(GC, TL)

    perform_source = [nmi_s, ari_s, pri_s, ps, rs, fs]
    perform_target = [nmi_t, ari_t, pri_t, pt, rt, ft]
    return perform_source, perform_target


def purity(Pred, Labels):
    N = len(set(Labels))
    M = len(set(Pred))
    matrix = np.zeros((M, N))
    for i, v in enumerate(Labels):
        q = Pred[i]
        matrix[q, v] += 1
    a = np.sum(matrix)
    b = (np.sum(matrix, axis=1) / a)
    p = np.max(matrix / np.sum(matrix, axis=1).reshape([-1, 1]), axis=1)
    return np.sum(p * b)


def PRF1(Pred, Labels):
    N = len(set(Labels))
    M = len(set(Pred))
    matrix = np.zeros((M, N))
    for i, v in enumerate(Labels):
        q = Pred[i]
        matrix[q, v] += 1
    TP = np.sum(matrix * (matrix - 1) / 2)
    a = (np.sum(matrix, axis=1))

    TFP = np.sum(a * (a - 1) / 2)
    FP = TFP - TP
    b = (np.sum(matrix, axis=0))

    FN = np.sum(b * (b - 1) / 2) - TP

    c = np.sum(a)
    TN = (c * (c - 1) / 2) - TP - FP - FN

    Prec = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Prec * Recall / (Prec + Recall)
    return Prec, Recall, F1


def measure_sequence(FL, GL, Ft=None, Gt=None, filename=None):
    fids = np.where(FL >= 0)[0]
    gids = np.where(GL >= 0)[0]
    print(fids, gids)

    new_FL = np.array(FL[fids], np.int).tolist()
    new_GL = np.array(GL[gids], np.int).tolist()

    if filename is not None:
        Ft, Gt = load_pkl(filename)

    infos = []
    for i in range(len(Ft)):
        F1 = np.unique(np.argmax(Ft[i][fids], axis=1), return_inverse=True)[1]
        G1 = np.unique(np.argmax(Gt[i][gids], axis=1), return_inverse=True)[1]

        result = measure(F1, G1, new_FL, new_GL)
        infos.append(result)
    return infos


def measure_sequence2(FL, GL, Ft=None, Gt=None, filename=None):
    FL = FL.reshape(-1)
    GL = GL.reshape(-1)
    fids = np.where(FL >= 0)[0]
    gids = np.where(GL >= 0)[0]
    # print(fids, gids, FL., GL)

    new_FL = np.array(FL[fids], np.int).tolist()
    new_GL = np.array(GL[gids], np.int).tolist()

    if filename is not None:
        Ft, Gt = load_pkl(filename)

    infos = []
    for i in range(len(Ft)):
        F1 = np.unique(np.argmax(Ft[i][fids], axis=1), return_inverse=True)[1]
        G1 = np.unique(np.argmax(Gt[i][gids], axis=1), return_inverse=True)[1]

        result = measure(F1, G1, new_FL, new_GL)
        infos.append(result)
    return infos


def measure_result(FL, GL, FC, GC):
    FL = FL.reshape(-1)
    GL = GL.reshape(-1)

    fids = np.where(FL >= 0)[0]
    gids = np.where(GL >= 0)[0]

    new_FL = np.array(FL[fids], np.int).tolist()
    new_GL = np.array(GL[gids], np.int).tolist()
    F1 = np.unique(FC[fids], return_inverse=True)[1].tolist()
    G1 = np.unique(GC[gids], return_inverse=True)[1].tolist()
    # print(F1, G1)
    result = measure(F1, G1, new_FL, new_GL)
    return result
