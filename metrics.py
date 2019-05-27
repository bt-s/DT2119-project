#!/usr/bin/python3

"""metrics.py: Script used for computing and visualizing performance metrics

Usage from CLI: $ python3 metrics.py <int> *-bn 
Where <int> is an integer to specify the max pooling window and '-bn' specifies
to select a model that was trained using batch normalization.

Part of a project for the 2019 DT2119 Speech and Speaker Recognition course at
KTH Royal Institute of Technology"""

__author__ = "Pietro Alovisi, Romain Deffayet & Bas Straathof"


import numpy as np
from sklearn.metrics import precision_score, recall_score

from keras.models import load_model

from collections import OrderedDict

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
 
import pickle


def f1_score(p, r):
    """Compute the F1 score

    Args:
        p (float): the precision
        r (flota): the recall

    Returns: (float) the F1 score
    """
    if p == 0 and r == 0:
        return 0
    else:
        return 2 * p * r / (p + r)


def compute_metric(labels, predictions, mtype="micro"):
    """Comutes the precision, recall and F1 scores

    Args:
        labels      (np.ndarray): the real labels
        predictions (np.ndarray): the model predictions
        mtype           (string): the type of metric; one of 'micro
                                  or 'macro'

    Returns:
        p  (float): the precision
        r  (float): the recall
        f1 (float): the F1 score
    """
    if mtype == 'micro':
        p  = precision_score(labels.flatten(), predictions.flatten())
        r  = recall_score(labels.flatten(), predictions.flatten())
        f1 = f1_score(p, r)

    elif mtype == 'macro':
        ps, rs, f1s = [], [], []
        for i, (label, pred) in enumerate(zip(labels, predictions)):
            p = precision_score(label, pred)
            r = recall_score(label, pred)
            ps.append(p), rs.append(r), f1s.append(f1_score(p, r))

        p, r, f1 = np.mean(ps), np.mean(rs), np.mean(f1s)

    return p, r, f1


def compute_metric_per_instrument(labels, predictions):
    """Comutes the precision, recall and F1 scores per instrument

    Args:
        labels      (np.ndarray): the real labels
        predictions (np.ndarray): the model predictions

    Returns:
        ps  (dict): the precision per instrument
        rs  (dict): the recall per instrument
        f1s (dict): the F1 scores per instrument
    """
    instruments = OrderedDict({"cel" : 0.0, "cla" : 0.0, "flu" : 0.0, "gac" : 0.0,
            "gel" : 0.0, "org" : 0.0, "pia" : 0.0, "sax" : 0.0,
            "tru" : 0.0, "vio" : 0.0, "voi" : 0.0})

    ps, rs, f1s = instruments.copy(), instruments.copy(), instruments.copy()
    for label, pred, inst in zip(labels.T, predictions.T, instruments):
        p = precision_score(label, pred)
        r = recall_score(label, pred)
        ps[inst] = p
        rs[inst] = r
        f1s[inst] = f1_score(p, r)

    return ps, rs, f1s


def main(argv):
    batch_norm = False
    max_pooling_window = int(argv[1])
    if len(argv) > 2 and argv[2] == "-bn":
        batch_norm = True
    else:
        raise Exception("The third input argument has to be '-bn'.")

    print("Max pooling window: ", max_pooling_window)
    print("Batch normlization: ", batch_norm)

    f = open("datasets/X_test.pkl", "rb")
    X_test = pickle.load(f)

    f = open("datasets/y_test.pkl", "rb")
    y_test = pickle.load(f)

    f.close()

    if batch_norm: model = load_model("results/with_batch_norm/model.h5")
    else: model = load_model("results/without_batch_norm/model.h5")

    predictions = np.zeros((len(X_test), 11))

    for (key,val) in X_test.items():
        val = np.reshape(val, (val.shape[0], 1, val.shape[1],
            val.shape[2]))

        prediction = model.predict(val)

        ### Class-wise max pooling:
        if max_pooling_window:
            max_pooled_pred = np.zeros_like(predictions)
            N = len(predictions)
            for i in range(N):
                max_pooled_pred[i] = np.max(predictions[max(0, i - max_pooling_window // 2) : 
                    min(N, i + max_pooling_window // 2), :], axis = 0)
            predictions = max_pooled_pred

        m = np.max(prediction, axis = 0)
        prediction = np.mean(prediction, axis = 0)
        prediction = prediction - 0.15
        prediction[prediction > 0] = 1
        prediction[prediction <= 0] = 0
        predictions[key] = prediction

    a = np.array(list(y_test.values()))

    y_test = np.zeros((len(a), 11))
    for i in range(len(a)):
        y_test[i] = a[i][0]

    micro_p, micro_r, micro_f1 = compute_metric(y_test, predictions)
    macro_p, macro_r, macro_f1 = compute_metric(y_test, predictions, mtype='macro')
    inst_p, inst_r, inst_f1 = compute_metric_per_instrument(y_test, predictions)

    print(micro_p, micro_r, micro_f1)
    print()
    print(macro_p, macro_r, macro_f1)
    print()
    print(inst_p, inst_r, inst_f1)

if __name__ == "__main__":
    main(sys.argv)

