import codecs
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import yaml
import ImageToSCC as imscc

from Tortuosity_Measures import TortuosityMeasures
from Morphology_Measures import Morphology_Measures as morph
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





def test_branch():
    with open('./config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    sp = (config_data["start_position"]["y"], config_data["start_position"]["x"])
    image_path = config_data["base_folder"] + config_data["binary_image"]

    assert os.path.isfile(image_path), "The image {} doesn't exixt".format(config_data["binary_image"])

    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    o_file = config_data["base_folder"] + config_data["output_file"]
    d_file = config_data["base_folder"] + config_data["distances_file"]

    

    treepath = imscc.build_tree(o_image, sp)
    interp_tree = imscc.build_interpolated_tree(treepath)

    tort = TortuosityMeasures.SCC_Tree(interp_tree)
    angle = morph.tree_branch_anlge(interp_tree)

    print("{} - Tort = {} - Angle = {}".format(config_data["binary_image"], tort, angle))
    print("{}\t{}\t{}".format(config_data["binary_image"], tort, angle))


def test_all():
    ###
    # 
    # ###

    # type_tree = "norm_trees"
    # folder = "norm_folder"
    type_tree = "hyper_trees"
    folder = "hyper_folder"

    with open('./positions.yaml', 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)

    trees = config_data[type_tree]        
    for tree in trees:
        image_path = config_data[folder] + tree["binary_image"]        
        assert os.path.isfile(image_path), "The image {} doesn't exixt".format(tree["binary_image"])

        o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        positons  = tree["start_position"]
        for elem in positons:
            sp = (elem["position"]["y"], elem["position"]["x"])

            treepath = imscc.build_tree(o_image, sp)
            interp_tree = imscc.build_interpolated_tree(treepath)

            tort = TortuosityMeasures.SCC_Tree(interp_tree)
            [m_angle, angles] = morph.tree_branch_anlge(interp_tree)

            print("{} - Tort = {} - Angle = {}".format(tree["binary_image"], tort, m_angle))
            # print("{}\t{}\t{}".format(config_data["binary_image"], tort, angle))
    

    




if __name__ == '__main__':
    # test_circle(1)
    # test_branch()
    test_all()