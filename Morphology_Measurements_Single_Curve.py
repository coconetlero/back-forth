import math
import numpy as np



def DM(curve):
    d = np.diff(curve, axis=0)
    segments = np.hypot(d[:, 0], d[:, 1])
    d_c = float(segments.sum())
    d_x = math.dist(curve[-1], curve[0])
    
    tort = (d_c / d_x) - 1.
    return tort



def SCC(curve):
    """
    Computes the Tortousity of a curve given by x_vec, y_vec with the Slope Chain Code method
    :param x_vec: numpy array X coordinates of the curve
    :param y_vec: numpy array Y coordinates of the curve
    :return the value of tortuosity given by the Slope Chain Code method
    """
    X = curve[:, 0]
    Y = curve[:, 1]
    diff_x = np.diff(X)
    diff_y = np.diff(Y)
    
    Theta = np.rad2deg(np.arctan2(diff_y, diff_x))
    alpha = np.diff(Theta)                

    trouble_idx = np.where(np.logical_or(alpha >= 180, alpha <= -180))
    trouble_values = alpha[trouble_idx]
    alpha[trouble_idx] = np.mod(trouble_values, np.sign(trouble_values) * (-360))
    alpha = alpha / 180

    SCC = np.sum(np.absolute(alpha))
    SCC_N = SCC / len(X)

    return [SCC, SCC_N]
    

def ArcLen(curve):
    d = np.diff(curve, axis=0)
    segments = np.hypot(d[:, 0], d[:, 1])
    arclen = float(segments.sum())
    return arclen