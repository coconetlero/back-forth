import codecs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Tortuosity_Measures import TortuosityMeasures
from numpy import genfromtxt, diff


matplotlib.use('Qt5Agg')

# # calculate scc on a csv file
# data = genfromtxt('C:/Users/zian/Projects/SCC-Trees/src_py/bb.csv', delimiter=',')
# x = data[:, 0]
# y = data[:, 1]
#
# diff_x = np.diff(x)
# diff_y = np.diff(y)
#
# Theta = np.rad2deg(np.arctan2(diff_y, diff_x))
# alpha = np.diff(Theta) / 180
# SCC = np.sum(np.absolute(alpha))
#
# print(SCC)

# -----------------------------------------------------------------

def plot_segment(X, Y):
    plt.plot(X, Y, 'r.-', linewidth=1, markersize=2)
    plt.axis('equal')
    plt.show()


## read a test tree and calculate SCC and tortuosity
def test_branches():
    lines = []
    with open('/Users/zianfanti/IIMAS/Three_Representation/data/test/test_branch.txt') as file:
        lines = file.readlines()

    branches = []
    branch = []

    for line in lines:
        line_content = line.split()
        coords = [float(line_content[0]), float(line_content[1])]

        if float(coords[0]) == -1.0 and float(coords[1]) == -1.0:
            if len(branch) != 0:
                branches.append(branch)
                b = np.array(branch)

            branch = []
            continue

        branch.append(coords)

    # count vertices in
    coords_num = 0
    for b in branches:
        coords_num = coords_num + len(b)

    # print slop change chains
    result_file = codecs.open('/Users/zianfanti/IIMAS/Three_Representation/data/test/t2.scc', "w+", "utf-8")
    result_file.write('{}\n'.format(coords_num))

    idx = 0
    for b in branches:
        bb = np.array(b)
        plot_segment(bb[:, 0], bb[:, 1])

        [SCC, alpha_f] = TortuosityMeasures.scc(bb[:, 0], bb[:, 1])  # forwards slope chain
        [SCC, alpha_b] = TortuosityMeasures.scc(bb[0:len(bb), 0][::-1], bb[:len(bb), 1][::-1])  # backwards slope chain

        for v in alpha_f:
            result_file.write('{:.6f} '.format(v))

        if idx > 0:
            result_file.write('1.0 ')
            for v in alpha_b[1:len(alpha_b)]:
                result_file.write('{:.6f} '.format(v))

        result_file.write('\n')

        idx += 1
        if idx == 3:
            break

    result_file.close()

    plt.show()


def test_circle(r):
    x = 0
    y = 0

    P = 2 * np.pi * r
    th = np.linspace(0, 2 * np.pi, 360)

    Xa = r * np.cos(th) + x
    Ya = r * np.sin(th) + y

    np.append(Xa, Xa[1])
    np.append(Ya, Ya[1])

    [SCC, alpha_f] = TortuosityMeasures.scc(Xa, Ya)
    print("SCC = {}".format(SCC))


    plot_segment(Xa, Ya)


if __name__ == '__main__':
    test_circle(1)