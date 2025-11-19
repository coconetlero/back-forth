import math
import numpy as np


def from_cont_to_discrete_curve(curve):
    X = curve[:,0]
    Y = curve[:,1]
    px = []
    py = []

    for x_i1, x_i2, y_i1, y_i2 in zip(X, X[1:], Y, Y[1:]):
        v1 = np.array(x_i1, y_i1)
        v2 = np.array(x_i2, y_i2)   
        distance = np.linalg.norm(v2 - v1)

        if distance > math.sqrt(2):
            print("Error")
        else:
            xt = round(x_i1)
            yt = round(y_i1)
            if xt != px[-1] or yt != py[-1]:
                px.append(xt)
                py.append(yt)

    return np.column_stack([np.array(px), np.array(py)])



