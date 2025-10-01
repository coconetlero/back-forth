import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml


import ImageToSCC as imscc
from SCC_Tree import SCC_Tree
from Morphology_Measures import Morphology_Measures
from Tortuosity_Measures import TortuosityMeasures

matplotlib.use('Qt5Agg')

def measure_neuron_tree(config_file, image_filename):
    with open(config_file, 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)
    im = None
    images = config_data['skeleton_images']
    for i, item in enumerate(images):
        if item['filename'] == image_filename:
            im = config_data['skeleton_images'][i]
            break

    assert im is not None, "file: " + image_filename + " doesn't exist"

    im = config_data['skeleton_images'][i]
    if im['subfolder'] is not None:
        image_path = os.path.join(config_data["image_folder"], im['subfolder'], im['filename'])
    else:
        image_path = os.path.join(config_data["image_folder"], im['filename'])


    start_position = im['start_position'][0]
    sp = (start_position['position']['y'],start_position['position']['x'])
    
    assert os.path.isfile(image_path), "The image {} doesn't exixt".format(config_data["binary_image"])

    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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

    print("{} \tTort = {} \tAngle = {}".format(im['filename'], T, m_angle))  

    return interp_tree    


def measure_all(config_file):

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

    with open('neuron_config.yaml', 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)
    
    images = config_data['skeleton_images']
    for i, item in enumerate(images):
        im = config_data['skeleton_images'][i]
        image_path = os.path.join(config_data["image_folder"], im['filename'])

        start_position = im['start_position'][0]
        sp = (start_position['position']['y'],start_position['position']['x'])
        
        assert os.path.isfile(image_path), "The image {} doesn't exixt".format(config_data["binary_image"])

        o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        treepath = imscc.build_tree(o_image, sp)
        interp_tree = imscc.build_interpolated_tree(treepath)
        [scc_tree, dist] = imscc.build_scc_tree(interp_tree)


        # o_file = config_data["base_folder"] + config_data["output_file"]
        # d_file = config_data["base_folder"] + config_data["distances_file"]

        # [X, Y] = imscc.plot_tree(scc_tree, dist)

        [T, T_n] = TortuosityMeasures.Tree_SCC(scc_tree)      
        [dm_m, dm_st, dm_t] = TortuosityMeasures.DM_Tree(interp_tree)      
        L = Morphology_Measures.tree_length(interp_tree)
        [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge(scc_tree)
        [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)
        [branch_mean_length, branch_sum_length, branch_lengths] = Morphology_Measures.tree_branch_length(interp_tree)
        T_c = Morphology_Measures.tree_scc_circularity(scc_tree)
        T_l = Morphology_Measures.tree_scc_linearity(scc_tree)
        [C, C_m] = Morphology_Measures.convex_concav(scc_tree)

        names_1.append(im['filename'])
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

        names_2 += [im['filename']] * len(t_angles)
        angles += t_angles

        names_3 += [im['filename']] * len(branch_lengths)
        all_branch_lengths += branch_lengths


        print("{} \tTort = {} \tAngle = {}".format(im['filename'], T, m_angle))        
            
    tp1 = list(zip(names_1, tort, lengths, branch_length, segments, bifurcations, terminals, dm_stl, circularity, linearity, conv, conv_m))
    tp2 = list(zip(names_2, angles)) 
    tp3 = list(zip(names_3, all_branch_lengths)) 

    df1 = pd.DataFrame(tp1, columns=['Image Name', 'Tortuosity', 'Length', 'Average Length', 'Segments', 'Bifurcations', 
                                     'Terminals', 'DM Tort', 'Circularity', 'Linearity', 'Convexity', 'Conv Mag'])
    df2 = pd.DataFrame(tp2, columns=['Image Name', 'Angles'])
    df3 = pd.DataFrame(tp3, columns=['Image Name', 'Branch Length'])

    df1.to_csv(config_data["masurements_csv_file"], mode='a')
    df2.to_csv(config_data["masurements_csv_file"], mode='a')
    df3.to_csv(config_data["masurements_csv_file"], mode='a')


def measure_all_v2(config_file):

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
    noncircularity = []
    norm_noncir = []
    linearity = []
    conv = []
    conv_m = []
    angles_mean = []
    mean_SSC = []

    with open(config_file, 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)
        
    images = config_data['skeleton_images']
    for i, item in enumerate(images):
        im = config_data['skeleton_images'][i]
        image_path = os.path.join(config_data["image_folder"], im['subfolder'], im['filename'])

        start_position = im['start_position'][0]
        sp = (start_position['position']['y'],start_position['position']['x'])
        
        assert os.path.isfile(image_path), "The image {} doesn't exixt".format(image_path)
        print("Processing image: {}".format(image_path))    

        o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        treepath = imscc.build_tree(o_image, sp)
        interp_tree = imscc.build_interpolated_tree(treepath)
        [scc_tree, dist] = imscc.build_scc_tree(interp_tree)


        # o_file = config_data["base_folder"] + config_data["output_file"]
        # d_file = config_data["base_folder"] + config_data["distances_file"]

        # [X, Y] = imscc.plot_tree(scc_tree, dist)

        [T, T_n] = TortuosityMeasures.Tree_SCC(scc_tree)      
        [dm_m, dm_st, dm_t] = TortuosityMeasures.DM_Tree(interp_tree)      
        L = Morphology_Measures.tree_length(interp_tree)
        [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge(scc_tree)
        [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)
        [branch_mean_length, branch_sum_length, branch_lengths] = Morphology_Measures.tree_branch_length(interp_tree)
        [T_c, Tc_norm] = Morphology_Measures.tree_scc_circularity(scc_tree)
        T_l = Morphology_Measures.tree_scc_linearity(scc_tree)
        [C, C_m] = Morphology_Measures.convex_concav(scc_tree)
        mean_scc = Morphology_Measures.slope_change_mean(scc_tree)
        
        names_1.append(im['filename'])
        tort.append(T)
        lengths.append(L)
        segments.append(seg)
        bifurcations.append(bifur)
        terminals.append(term)
        branch_length.append(branch_mean_length)
        dm_stl.append(dm_st)
        noncircularity.append(T_c)
        norm_noncir.append(Tc_norm)
        linearity.append(T_l)
        conv.append(C)
        conv_m.append(C_m)
        angles_mean.append(m_angle)
        mean_SSC.append(mean_scc)
        

        names_2 += [im['filename']] * len(t_angles)
        angles += t_angles

        names_3 += [im['filename']] * len(branch_lengths)
        all_branch_lengths += branch_lengths


        print("{} \tTort = {} \tAngle = {}".format(im['filename'], T, m_angle))        
            
    tp1 = list(zip(names_1, tort, lengths, branch_length, segments, bifurcations, terminals, dm_stl, noncircularity, norm_noncir, linearity, conv, conv_m, angles_mean, mean_SSC))
    tp2 = list(zip(names_2, angles)) 
    tp3 = list(zip(names_3, all_branch_lengths)) 

    df1 = pd.DataFrame(tp1, columns=['Image Name', 'Tortuosity', 'Length', 'Average Length', 'Segments', 'Bifurcations',
                                     'Terminals', 'DM Tort', 'NonCircularity', 'NonCircularity (Normalized)', 'Linearity', 'Convexity', 'Conv Mag', 'Angle (Mean)', 'Mean SCC'])
    df2 = pd.DataFrame(tp2, columns=['Image Name', 'Angles'])
    df3 = pd.DataFrame(tp3, columns=['Image Name', 'Branch Length'])

    if os.path.exists(config_data["masurements_csv_file"]):
        os.remove(config_data["masurements_csv_file"])

    df1.to_csv(config_data["masurements_csv_file"], mode='a')
    df2.to_csv(config_data["masurements_csv_file"], mode='a')
    df3.to_csv(config_data["masurements_csv_file"], mode='a')



def test_tree_class(config_file):

    with open(config_file, 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)
        
    images = config_data['skeleton_images']
    for i, item in enumerate(images):
        im = config_data['skeleton_images'][i]
        image_path = os.path.join(config_data["image_folder"], im['subfolder'], im['filename'])

        start_position = im['start_position'][0]
        sp = (start_position['position']['y'],start_position['position']['x'])
        
        assert os.path.isfile(image_path), "The image {} doesn't exixt".format(image_path)
        print("Processing image: {}".format(image_path))    

        o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        treepath = imscc.build_tree(o_image, sp)
        interp_tree = imscc.build_interpolated_tree(treepath)
        [scc_tree, dist] = imscc.build_scc_tree(interp_tree)

        # imscc.display_tree(scc_tree, dist)
        # [X, Y] = imscc.plot_tree(scc_tree, dist)

        scc_tree = SCC_Tree(interp_tree)    
        scc_tree_o = []
        for e in scc_tree.tree:
            if e <= 1:
                scc_tree_o.append(e)

        ext_tree = SCC_Tree.create_extended_tree_3(scc_tree_o)
        ext_tree.insert(2, 0)
        SCC_Tree.plot_tree(ext_tree)



if __name__ == '__main__':
    # measure_neuron_tree('neuron_config_2.yaml', 'AdGoI-neuronD.CNG.tif')

    # measure_all()

    # measure_all_v2('neuron_config_2.yaml')

    test_tree_class('neuron_config_2.yaml')

