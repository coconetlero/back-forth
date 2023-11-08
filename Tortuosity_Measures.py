import math

import numpy as np


class TortuosityMeasures:

    def scc(x_vec, y_vec):
        """
        Computes the Tortousity of a curve given by x_vec, y_vec with the Slope Chain Code method
        :param x_vec: numpy array X coordinates of the curve
        :param y_vec: numpy array Y coordinates of the curve
        :return the value of tortuosity given by the Slope Chain Code method
        """
        diff_x = np.diff(x_vec)
        diff_y = np.diff(y_vec)

        Theta = np.rad2deg(np.arctan2(diff_y, diff_x))
        alpha = np.diff(Theta) / 180
        alpha = np.insert(alpha, 0, Theta[0]/180)
        SCC = np.sum(np.absolute(alpha))

        return [SCC, alpha]
