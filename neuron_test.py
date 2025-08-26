import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

import ImageToSCC as imscc
from Morphology_Measures import Morphology_Measures
from Tortuosity_Measures import TortuosityMeasures

matplotlib.use('Qt5Agg')

def measure_neuron_tree():
    with open('/Users/zianfanti/Trabajo/tree_representation/back-forth/neuron_config.yaml', 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)

    im = config_data['skeleton_images'][0]
    image_path = os.path.join(config_data["image_folder"], im['filename'])

    start_position = im['start_position'][0]
    sp = (start_position['position']['y'],start_position['position']['x'])
    
    assert os.path.isfile(image_path), "The image {} doesn't exixt".format(config_data["binary_image"])

    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # o_file = config_data["base_folder"] + config_data["output_file"]
    # d_file = config_data["base_folder"] + config_data["distances_file"]

    treepath = imscc.build_tree(o_image, sp)
    interp_tree = imscc.build_interpolated_tree(treepath)

    [scc_tree, dist] = imscc.build_scc_tree(interp_tree)

    imscc.display_tree(scc_tree, dist)
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

    return interp_tree    

if __name__ == '__main__':
    measure_neuron_tree()