import codecs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Tortuosity_Measures import TortuosityMeasures
from numpy import genfromtxt, diff

# matplotlib.use('Qt5Agg')        # for windows OS
matplotlib.use('MacOSX')

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
    plt.plot(X, Y, 'r.-')
    plt.axis('equal')


if __name__ == '__main__':
    ## parsing the file and crete branche

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
