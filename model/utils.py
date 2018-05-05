import numpy as np


def softmax(x, temp):
    return np.exp(x / temp) / np.sum(np.exp(x / temp))
