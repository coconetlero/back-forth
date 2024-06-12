
import cv2
import os
import pandas as pd
import yaml

import ImageToSCC as imscc

from SCC_Tree import SCC_Tree
from Tortuosity_Measures import TortuosityMeasures
from Morphology_Measures import Morphology_Measures


def measure_branch_length(configuration):
    all_branch_lengths = []
    disease = ""
    for type_tree in ["norm_trees", "hyper_trees"]:        
        if type_tree == "norm_trees" : 
            folder = "norm_folder"
            disease = "Normal"
        if type_tree == "hyper_trees" : 
            folder = "hyper_folder"
            disease = "Hypertension"

        type_trees = configuration[type_tree]        
        for container_image in type_trees:
            image_path = configuration[folder] + container_image["binary_image"]        
            assert os.path.isfile(image_path), "The image {} doesn't exist".format(container_image["binary_image"])

            o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            trees = container_image["start_position"]

            for tree_root in trees:
                sp = (tree_root["position"]["y"], tree_root["position"]["x"])

                treepath = imscc.build_tree(o_image, sp)
                interp_tree = imscc.build_interpolated_tree(treepath)

                [branch_mean_length, branch_sum_length, branch_lengths] = Morphology_Measures.tree_branch_length(interp_tree)
                
                lengths =  [[container_image["binary_image"], str(container_image["type"]).capitalize(), disease, l] for l in branch_lengths]
                all_branch_lengths += lengths
        
    df = pd.DataFrame(all_branch_lengths, columns=['Image Name', 'Vessel', 'Disease', 'Branch Length'])
    if os.path.isfile(configuration["branch_length_file"]):
        os.remove(configuration["branch_length_file"])
    df.to_csv(configuration["branch_length_file"], mode='w')


def measure_branch_angles(configuration):
    all_angles = []
    disease = ""
    for type_tree in ["norm_trees", "hyper_trees"]:        
        if type_tree == "norm_trees" : 
            folder = "norm_folder"
            disease = "Normal"
        if type_tree == "hyper_trees" : 
            folder = "hyper_folder"
            disease = "Hypertension"

        type_trees = configuration[type_tree]        
        for container_image in type_trees:
            image_path = configuration[folder] + container_image["binary_image"]        
            assert os.path.isfile(image_path), "The image {} doesn't exist".format(container_image["binary_image"])

            o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            trees = container_image["start_position"]

            for tree_root in trees:
                sp = (tree_root["position"]["y"], tree_root["position"]["x"])

                treepath = imscc.build_tree(o_image, sp)
                interp_tree = imscc.build_interpolated_tree(treepath)
                scc_tree= SCC_Tree(interp_tree).tree

                [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge(scc_tree)
                
                angles =  [[container_image["binary_image"], str(container_image["type"]).capitalize(), disease, l] for l in t_angles]
                all_angles += angles
        
    df = pd.DataFrame(all_angles, columns=['Image Name', 'Vessel', 'Disease', 'Angle'])
    if os.path.isfile(configuration["angles_file"]):
        os.remove(configuration["angles_file"])
    df.to_csv(configuration["angles_file"], mode='w')


def full_tree_measurements(configuration):
    measurements = []
    disease = ""
    for type_tree in ["norm_trees", "hyper_trees"]:        
        if type_tree == "norm_trees" : 
            folder = "norm_folder"
            disease = "Normal"
        if type_tree == "hyper_trees" : 
            folder = "hyper_folder"
            disease = "Hypertension"

        type_trees = configuration[type_tree]        
        for container_image in type_trees:
            image_path = configuration[folder] + container_image["binary_image"]        
            assert os.path.isfile(image_path), "The image {} doesn't exist".format(container_image["binary_image"])

            o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            trees = container_image["start_position"]

            for tree_root in trees:
                sp = (tree_root["position"]["y"], tree_root["position"]["x"])

                treepath = imscc.build_tree(o_image, sp)
                interp_tree = imscc.build_interpolated_tree(treepath)
                scc_tree= SCC_Tree(interp_tree).tree

                # [X, Y] = imscc.plot_tree(scc_tree, dist)

                [T, T_n] = TortuosityMeasures.Tree_SCC(scc_tree)      
                [dm_m, dm_st, dm_t] = TortuosityMeasures.DM_Tree(interp_tree)      
                L = Morphology_Measures.tree_length(interp_tree)
                [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge(scc_tree)
                [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)
                [branch_mean_length, branch_sum_length, branch_lengths] = Morphology_Measures.tree_branch_length(interp_tree)
                T_c = Morphology_Measures.tree_scc_circularity_2(scc_tree)
                T_l = Morphology_Measures.tree_scc_linearity(scc_tree)
                [C, C_m] = Morphology_Measures.convex_concav(scc_tree)

                tree_measure = [[container_image["binary_image"], disease, str(container_image["type"]).capitalize(),
                                 T, L, T / L, branch_mean_length, seg, bifur, term, dm_st, T_c, T_l, C_m]]
                measurements += tree_measure 

    df = pd.DataFrame(measurements, columns=['Image Name', 'Disease', 'Vessel', 'Tortuosity_SCC', 'Length',
                                             'T/L', 'Branch Length', 'Segments', 'Bifurcations', 'Terminals',
                                             'DM', 'Circularity', 'Linearity', 'Convex'])
    if os.path.isfile(configuration["measurements_file"]):
        os.remove(configuration["measurements_file"])
    df.to_csv(configuration["measurements_file"], mode='w')    

        

if __name__ == '__main__':

    # angles_file:  /Users/zianfanti/IIMAS/Three_Representation/results/db_Elena/angles_results.csv 
    # branch_length_file: /Users/zianfanti/IIMAS/Three_Representation/results/db_Elena/branch_length.csv
    # measurements_file:  /Users/zianfanti/IIMAS/Three_Representation/results/db_Elena/all_trees_measurements.csv

    with open('/Users/zianfanti/IIMAS/Three_Representation/src/back-forth/positions.yaml', 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)


    measure_branch_length(config_data)
    measure_branch_angles(config_data)
    full_tree_measurements(config_data)
