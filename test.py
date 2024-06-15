import codecs
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import yaml

import ImageToSCC as imscc

from SCC_Tree import SCC_Tree
from Morphology_Measures import Morphology_Measures
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



def test_bifurcation_finding():
    # scc_tree_o =  [1, 0, -.25, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.25, 0]    
    scc_tree =  [2, 1, 0, 0, 3, -.25, 0, 4, 1, 0, 3, -.75, 0, 5, 1, 0, 3, -.75, 0, 6, 1, 0, 3, -.25, 0]    
    dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 6, 1, 1, 3, 1, 1]

    # scc_tree_o =  [1, 0, -.5, 0, 1, 0, -.5, 0, 1, 0, 0, 0]    
    # scc_tree =  [2, 1, 0, 0, 3, -.5, 0, 4, 1, 0, 3, -.5, 0, 5, 1, 0, 0, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1]

    # # scc_tree_o =  [1, 0, -.1, 0, 1, 0, -.1, 0, 1, 0, -.8, 0]    
    # scc_tree =  [2, 1, 0, 0, 3, -.1, 0, 4, 1, 0, 3, -.1, 0, 5, 1, 0, 3, -0.8, 0, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 1]

    # # scc_tree_o =  [1, 0, .5, 0, 1, 0, -.75, 0, 1, 0, -.75, 0]    
    # scc_tree =  [2, 1, 0, 0, 3, .5, 0, 4, 1, 0, 3, -.75, 0, 5, 1, 0, 3, -0.75, 0, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 1]

    # # scc_tree_o =  [1, 0, 0, 0, 1, 0, .25, 0, 1, 0, .75, 0]    
    # scc_tree =  [2, 1, 0, 0, 0, 0, 3, 1, 0, 4, -.25, 0, 5, 1, 0, 4, -.75, 0]    
    # dist =      [2, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 5, 1, 1, 4, 1, 1]

    # # scc_tree_o =  [1, 0, -.25, 0, 1, 0, -.75, 0, 1, 0, -.25, 0, 1, 0, .25, 0]    
    # scc_tree =  [2, 1, 0, 0, 3, -.75, 0, 4, 1, 0, 3, -.25, 0, 5, 1, 0, 3, -.25, 0, 6, 1, 0, 3, .25, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 6, 1, 1, 3, 1, 1]

    # # scc_tree_o =  [1, 0, -.25, 0, 1, 0, -.5, 0, 1, 0, -.25, 0]    
    # scc_tree =  [2, 1, 0, 0, 3, -.25, 0, 4, 1, 0, 3, -.5, 0, 5, 1, 0, 3, -0.25, 0, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 1]

    # # scc_tree_o =  [1, 0, -.75, 0, 1, 0, -.25, 0, 1, 0, -.25, 0, 1, 0, .25, 0]   
    # scc_tree =  [2, 1, 0, 0, 3, -.75, 0, 4, 1, 0, 3, -.25, 0, 5, 1, 0, 3, -.25, 0, 6, 1, 0, 3, .25, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 6, 1, 1, 3, 1, 1]


    # scc_tree_o =    [1, 0, -.75, 0, 1, 0, -.25, 0, 0, 0, -.25, 0, 1, 0, -.5, 0, 1, 0, -.25, 0, 0, 0, -.25, 0, 1, 0, .25, 0]    
    # scc_tree =      [2, 1, 0, 0, 3, -.75, 0, 4, 1, 0, 3, -.25, 0, 0, 5, -.25, 0, 6, 1, 0, 5, -.5, 0, 7, 1, 0, 5, -.25, 0,  0, 3, -.25, 0, 8, 1, 0, 3, .25, 0]    
    # dist =          [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 1, 5, 1, 1, 6, 1, 1, 5, 1, 1, 7, 1, 1, 5, 1, 1, 1, 3, 1, 1, 8, 1, 1, 3, 1, 0]

    # scc_tree =  [2, 1, 0, 0, 3, -.75, 4, 1, 3, -.75, 5, 1, 3, -.75, 6, 1, 3, -.75, 7, 1, 3, -.75, 8, 1, 3, -.75, 9, 1, 3, -.75, 10, 1, 3, -.75, 0, 0]    
    # dist =      [2, 1, 1, 1, 3, 1, 4, 1, 3, 1, 5, 1, 3, 1, 6, 1, 3, 1, 7, 1, 3, 1, 8, 1, 3, 1, 9, 1, 3, 1, 10, 1, 3, 1, 1, 1]


    # scc_tree_o =    [1, 0, -.25, 0, -.5, 0, 0, 0, 1, 0, -.5, 0, 0, 1, 0, 0, -.5, 0, -.5, 0, 0, -.5, 1, -.5, 0, 1, 0, 0, 0, 0, 0, 0, -.5, 0, 0, 1, 0, 0, -.25, 0]    
    # scc_tree =      [2, 1, 0, 0, 3, -.25, 0, 4, -.5, 0, 0, 0, 5, 1, 0, 6, -.5, 0, 0, 7, 1, 0, 0, 6, -.5, 0, 4, -.5, 0, 0, 8, -.5, 9, 1, 8, -.5, 0, 10, 1, 0, 0, 0, 0, 0, 0, 3, -.5, 0, 0, 11, 1, 0, 0, 3, -.25, 0]    
    # dist =          [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 6, 1, 1, 4, 1, 1, 1, 8, 1, 9, 1, 8, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 11, 1, 1, 1, 3, 1, 0]


    # scc_tree_o =    [1, 0, 0, 0, -.5, 0, 1, 0, -.5, 0, 1, 0, -.5, 0, 1, 0, -.5, 0, 0, 0]    
    # scc_tree =      [2, 1, 0, 0, 0, 0, 3, -.5, 0, 4, 1, 0, 3, -.5, 0, 5, 1, 0, 3, -.5, 0, 6, 1, 0, 3, -.5, 0, 0, 0]    
    # dist =          [2, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 6, 1, 1, 3, 1, 1, 1, 1]

    scc_tree_o =    [1, 0, 0, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 1, 0, -.75, 0, 0, 0]    
    scc_tree =      [2, 1, 0, 0, 0, 0, 3, -.75, 0, 4, 1, 0, 3, -.75, 0, 5, 1, 0, 3, -.75, 0, 6, 1, 0, 3, -.75, 0, 7, 1, 0, 3, -.75, 0, 8, 1, 0, 3, -.75, 0, 9, 1, 0, 3, -.75, 0, 10, 1, 0, 3, -.75, 0, 0, 0]    
    dist =          [2, 1, 1, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 6, 1, 1, 3, 1, 1, 7, 1, 1, 3, 1, 1, 8, 1, 1, 3, 1, 1, 9, 1, 1, 3, 1, 1, 10, 1, 1, 3, 1, 1, 1, 1]

    # scc_tree_o =    [1,0,-.25,0,1,0,-.5,0,1,0,-.25,0]

    # imscc.display_tree(scc_tree, dist)
    [X, Y] = imscc.plot_tree(scc_tree, dist)
    tree_string = [2]
        
    push = False  
    stack = []
    idx_stack = []
    bifurcation_idx = {}
    for idx, e in enumerate(scc_tree_o):       
        
        if e == 1:
            push = not push
        else:                   
            if push:
                stack.append(e) 
                idx_stack.append(idx) 
            else:
                if stack[-1] == -e:
                    stack.pop()  
                    idx_stack.pop()                  
                else:                    
                    push = not push                    
                    if bifurcation_idx.get(idx_stack[-1]):
                        bifurcation_idx[idx_stack[-1]] = bifurcation_idx[idx_stack[-1]] + 1
                    else:
                        bifurcation_idx[idx_stack[-1]] = 1
                    stack.append(stack.pop() + e)
                    # if abs(stack[-1]) == 1.0 and bifurcation_idx[idx_stack[-1]] > 1:
                    if stack[-1].is_integer() and bifurcation_idx[idx_stack[-1]] > 1:
                        push = not push
                        stack.pop()
                        idx_stack.pop()

    print(bifurcation_idx)                 
                        

              

        




def test_circle(r):
    x = 0
    y = 0

    P = 2 * np.pi * r
    th = np.linspace(0, 2 * np.pi, 90)

    Xa = r * np.cos(th) + x
    Ya = r * np.sin(th) + y

    Xa = np.append(Xa, Xa[1])
    Ya = np.append(Ya, Ya[1])

    [SCC, alpha_f] = TortuosityMeasures.SCC(Xa, Ya)    
    
    scc_curve = imscc.create_scc_closed_curve(Xa, Ya)

    non_circ = Morphology_Measures.curve_scc_circularity(scc_curve)

    print("SCC = {}".format(SCC))
    print("Circ = {}".format(non_circ))
    plot_segment(Xa, Ya)





def test_paper():

    # scc_tree = [2, 1, 0, 0, 0, 0, 3, -.3, 0, -.1, 0, 4, 1, 0, .1, 0, 3, -.6, 0, 0, 5, -.2, -.3, 6, 1, 
    #             .3, 5, -.7, .2, 7, 1, -.2, 5, -.1, 0, 0, 3, -.8, 0, 0, .3, 8, 1, -.3, 0, 0, 3, -.3, 0, 0, 0, 0]
    # dist = [2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 3, 1, 1, 1, 5, 1, 1, 6, 1, 
    #         1, 5, 1, 1, 7, 1, 1, 5, 1, 1, 1, 3, 1, 1, 1, 1, 8, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1]
    
    scc_tree = [2, 1, 0, 0, 0, 3, -.3, 0, -.1, 0, 4, 1, 0, .1, 0, 3, -.6, 0, 0, 5, -.2, -.3, 6, 1, 
                .3, 5, -.7, .2, 7, 1, -.2, 5, -.1, 0, 0, 3, -.1, 0, 0, 0]
    dist = [2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 3, 1, 1, 1, 5, 1, 1, 6, 1, 
                1, 5, 1, 1, 7, 1, 1, 5, 1, 1, 1, 3, 1, 1, 1, 1]
    
    # scc_tree = [2, 1, 0, 3, 0.34, 0, 4, 1, 0, 3, 0.48, 0, 5, 1, 0, 3, 0.18, 0]
    # dist = [2, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1]

    # scc_tree = [2, 1, 0, 0, 3, -0.75, 0, 4, 1, 0, 3, 0.25, 0, 5, 1, 0, 3, -0.5, 0, 0]
    # dist = [2, 1, 1, 1, 3, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 3, 1, 1, 1]
    
    [X, Y] = imscc.plot_tree(scc_tree, dist)
    # imscc.display_tree(scc_tree, dist)

    
    # [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge(scc_tree)
    [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)
    # non_circ = Morphology_Measures.tree_scc_circularity(scc_tree)
    # C = Morphology_Measures.convex_concav(scc_tree)


def open_tree_def():
    with open('/Users/zianfanti/IIMAS/Three_Representation/src/back-forth/config.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

    sp = (config_data["start_position"]["y"], config_data["start_position"]["x"])
    image_path = config_data["base_folder"] + config_data["binary_image"]

    assert os.path.isfile(image_path), "The image {} doesn't exixt".format(config_data["binary_image"])

    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    o_file = config_data["base_folder"] + config_data["output_file"]
    d_file = config_data["base_folder"] + config_data["distances_file"]

    treepath = imscc.build_tree(o_image, sp)
    interp_tree = imscc.build_interpolated_tree(treepath)

    return interp_tree    
    

def test_branch():

    interp_tree = open_tree_def()
    [scc_tree, dist] = imscc.build_scc_tree(interp_tree)

    # imscc.display_tree(scc_tree, dist)
    [X, Y] = imscc.plot_tree(scc_tree, dist)

    # [dm_m, dm_st, dm_t] = TortuosityMeasures.DM_Tree(interp_tree)      
    # tort = TortuosityMeasures.SCC_Tree(scc)
    # [m_angle, angles] = morph.tree_scc_branch_anlge(scc)
    # [segments, bifurcations, terminals] = Morphology_Measures.tree_scc_count_features(scc)
    # lengths = Morphology_Measures.tree_branch_length(interp_tree)
    non_circ = Morphology_Measures.tree_scc_circularity(scc_tree)
    # [mt, at, t] = TortuosityMeasures.DM_Tree(interp_tree)
    # T_l = Morphology_Measures.tree_scc_linearity(scc_tree)
    # C = Morphology_Measures.convex_concav(scc_tree)

    # print("{} - Tort = {} - Angle = {}".format(config_data["binary_image"], tort, angle))
    # print("{}\t{}\t{}".format(config_data["binary_image"], tort, angle))



def test_all():
    ###
    # 
    # ###
    with open('/Users/zianfanti/IIMAS/Three_Representation/src/back-forth/positions.yaml', 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)

    type_tree = "norm_trees"
    folder = "norm_folder"
    result_file = "n_csv_output"
    
    # type_tree = "hyper_trees"
    # folder = "hyper_folder"
    # result_file = "h_csv_output"


    names_1 = []
    names_2 = []
    names_3 = []
    tort = []
    angles = []
    lengths = []
    segments = []
    bifurcations = []
    terminals = []
    branch_length = []
    all_branch_lengths = []
    dm_stl = []
    circularity = []
    linearity = []
    conv = []
    conv_m = []
    
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
            [scc_tree, dist] = imscc.build_scc_tree(interp_tree)

            [X, Y] = imscc.plot_tree(scc_tree, dist)

            [T, T_n] = TortuosityMeasures.Tree_SCC(scc_tree)      
            [dm_m, dm_st, dm_t] = TortuosityMeasures.DM_Tree(interp_tree)      
            L = Morphology_Measures.tree_length(interp_tree)
            [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge(scc_tree)
            [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)
            [branch_mean_length, branch_sum_length, branch_lengths] = Morphology_Measures.tree_branch_length(interp_tree)
            T_c = Morphology_Measures.tree_scc_circularity(scc_tree)
            T_l = Morphology_Measures.tree_scc_linearity(scc_tree)
            [C, C_m] = Morphology_Measures.convex_concav(scc_tree)

            names_1.append(tree["binary_image"])
            tort.append(T)
            lengths.append(L)
            segments.append(seg)
            bifurcations.append(bifur)
            terminals.append(term)
            branch_length.append(branch_mean_length)
            dm_stl.append(dm_st)
            circularity.append(T_c)
            linearity.append(T_l)
            conv.append(C)
            conv_m.append(C_m)

            names_2 += [tree["binary_image"]] * len(t_angles)
            angles += t_angles

            names_3 += [tree["binary_image"]] * len(branch_lengths)
            all_branch_lengths += branch_lengths


            # print("{} \tTort = {} \tAngle = {}".format(tree["binary_image"], tort, m_angle))
            # print("{}\t{}\t{}".format(config_data["binary_image"], tort, angle))
            
    tp1 = list(zip(names_1, tort, lengths, branch_length, segments, bifurcations, terminals, dm_stl, circularity, linearity, conv, conv_m))
    tp2 = list(zip(names_2, angles)) 
    tp3 = list(zip(names_3, all_branch_lengths)) 

    df1 = pd.DataFrame(tp1, columns=['Image Name', 'Tortuosity', 'Length', 'Average Length', 'Segments', 'Bifurcations', 
                                     'Terminals', 'DM Tort', 'Circularity', 'Linearity', 'Convexity', 'Conv Mag'])
    df2 = pd.DataFrame(tp2, columns=['Image Name', 'Angles'])
    df3 = pd.DataFrame(tp3, columns=['Image Name', 'Branch Length'])

    df1.to_csv(config_data[result_file], mode='a')
    df2.to_csv(config_data[result_file], mode='a')
    df3.to_csv(config_data[result_file], mode='a')
    

    

def test_tree_class():
    interp_tree = open_tree_def()
    scc_tree = SCC_Tree(interp_tree)

    T_c = Morphology_Measures.tree_scc_circularity_2(scc_tree)



if __name__ == '__main__':
    # test_circle(1)
    # test_paper()
    # test_branch()
    # test_all()
    # test_tree_class()
    test_bifurcation_finding()
    