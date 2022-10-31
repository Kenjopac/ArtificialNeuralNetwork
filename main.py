from palmerpenguins import load_penguins
import pandas as pd
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
import numpy
import math

def throwaway():
    for i in X_train.values.tolist():
        print(i)
        UnactivatedLayerOne, ActivatedLayerOne = forwardprop(i, w1, b1)
        print(UnactivatedLayerOne, ActivatedLayerOne)


def setup():
    w1 = [[random.random() for x in range(4)] for i in range(4)]
    b1 = [random.randint(0, 5) for i in range(4)]
    w2 = [[random.random() for x in range(4)] for i in range(3)]
    b2 = [random.randint(0, 5) for i in range(3)]
    return w1, b1, w2, b2

def sigmoid(x):
    return [1.0 / (1.0 + math.exp(-c)) for c in x]

def softmax(x):
    return [math.exp(t) / sum([math.exp(a) for a in x]) for t in x]
def relu(x):
    return [max(0,a) for a in x]

def forwardprop(input, weights, bias, activationfunc):  # input as 1d array, weights as list<array> of neuron weights, bias 1d
    TwoDlist = []
    for j in range(len(weights)):
        OneDlist = []
        for n in range(len(input)):
            OneDlist.append(weights[j][n] * input[n])
        TwoDlist.append(OneDlist)
    TwoDlist = [sum(a) for a in TwoDlist]
    Unactivatedlist = [TwoDlist[i] - bias[i] for i in range(len(TwoDlist))]
    ActivatedList = activationfunc(Unactivatedlist)
    return Unactivatedlist, ActivatedList

def meanSquaredError(predicted, observed):
    return [math.pow(predicted[i] - observed[i] for i in range(len(observed), 2))]
def AverageCost(predictedlist,Observedlist):
    return sum(meanSquaredError(predictedlist,Observedlist)) / float(len(predictedlist))

if __name__ == "__main__":
    penguins = load_penguins()
    x, y = load_penguins(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=100, random_state=0)
    w1, b1, w2, b2 = setup()
    UnactivatedLayerOne, ActivatedLayerOne = forwardprop(X_train.values.tolist()[0], w1, b1, sigmoid)
    UnactivatedLayerTwo, ActivatedLayertwo = forwardprop(ActivatedLayerOne, w2, b2, softmax)
    print(UnactivatedLayerOne, ActivatedLayerOne, UnactivatedLayerTwo, ActivatedLayertwo)