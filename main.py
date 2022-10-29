from palmerpenguins import load_penguins
import pandas as pd
import seaborn as sns
import random
from sklearn.model_selection import train_test_split

import math


def setup():
    w1 = [[random.random() for x in range(4)] for i in range(4)]
    b1 = [random.randint(0, 5) for i in range(4)]
    w2 = [[random.random() for x in range(4)] for i in range(3)]
    b2 = [random.randint(0, 5) for i in range(3)]
    return w1, b1, w2, b2

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def forwardprop(input, weights, bias):  # input as 1d array, weights as list<array> of neuron weights, bias 1d
    TwoDlist = []
    for j in range(len(weights)):
        OneDlist = []
        for n in range(len(input)):
            OneDlist.append(weights[j][n] * input[n])
        TwoDlist.append(OneDlist)
    TwoDlist = [sum(a) for a in TwoDlist]
    Unactivatedlist = []
    for i in range(len(TwoDlist)):
        Unactivatedlist.append(TwoDlist[i] - bias[i])
    ActivatedList = [sigmoid(float(t)) for t in Unactivatedlist]
    return Unactivatedlist, ActivatedList


if __name__ == "__main__":
    w1 = [0.25, 0.4, 0.3, 0.63]
    penguins = load_penguins()
    x, y = load_penguins(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=100, random_state=0)
    w1, b1, w2, b2 = setup()
    for i in X_train.values.tolist():
        print(i)
        UnactivatedLayerOne, ActivatedLayerOne = forwardprop(i, w1, b1)
        print(UnactivatedLayerOne, ActivatedLayerOne)
