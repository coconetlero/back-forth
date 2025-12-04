
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import re

import ImageToSCC as imscc


def save_pixelated_curve_set(curves_dir_path, description_filename, image_folder, curves_folder_path):    
    with open(os.path.join(curves_dir_path, description_filename), 'r', encoding='utf-8') as f:        
        for idx, line in enumerate(f):            
            match1 = re.search(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', line)               
            if match1:
                fname = match1.group(1)      
                name, _ = os.path.splitext(fname)         
                    
                x = float(match1.group(2))
                y = float(match1.group(3))
                sp = (int(y), int(x))

                o_image = cv2.imread(os.path.join(curves_dir_path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)   

                k = 2
                branch = []                
                while type(treepath[k]) is tuple:
                    branch.append(treepath[k])                    
                    k += 1
                
                px = [point[0] for point in branch]
                py = [point[1] for point in branch]
                pixel_curve = np.column_stack([px, py])                
                np.savetxt(os.path.join(curves_dir_path, curves_folder_path, name + ".txt"), pixel_curve, fmt="%s")                



def load_curve_from_txt_file(file_path):
    """
    Load a curve from a txt file
    """
    curve = np.loadtxt(file_path, dtype=np.float32)
    return curve


def load_curves_from_txt_file(folder_path):
    """
    Load all curves from a folder containing txt files
    """
    curves = []
    filenames = []
    filtered_filenames = [item for item in os.listdir(folder_path) if not item.startswith('._')]
    for filename in filtered_filenames:
        file_path = os.path.join(folder_path, filename)
        curve = load_curve_from_txt_file(file_path)
        curves.append(curve)
        filenames.append(filename)
    
    return curves, filenames



def plot_results(curve1, curve2):
    """
    Plot the results of the polynomial fitting
    """
    plt.figure(figsize=(12, 12))
    plt.plot(curve1[:, 0], curve1[:, 1], 'bo-', alpha=0.3, markersize=2, linewidth=1, label='Original')
    plt.plot(curve2[:, 0], curve2[:, 1], 'ro-', alpha=0.8, markersize=2, linewidth=1, label='Smoothed')


    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Curve Smoothing with Cubic Spline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()