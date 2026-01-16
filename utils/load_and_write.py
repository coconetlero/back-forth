
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
                    
                x = int(match1.group(2))
                y = int(match1.group(3))
                sp = (y, x)

                o_image = cv2.imread(os.path.join(curves_dir_path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)   
 
                k = 2
                branch = []                
                while type(treepath[k]) is tuple:
                    branch.append(treepath[k])                    
                    k += 1
                
                px = [point[1] for point in branch]
                py = [point[0] for point in branch]
                pixel_curve = np.column_stack([px, py])                
                np.savetxt(os.path.join(curves_dir_path, curves_folder_path, name + ".txt"), pixel_curve, fmt="%s")                



def load_float_curve_from_txt_file(file_path):
    """
    Load a curve from a txt file
    """
    curve = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
    return curve


def load_float_curves_from_txt_files(folder_path):
    """
    Load all curves from a folder containing txt files
    """
    curves = []
    filenames = []
    filtered_filenames = [item for item in os.listdir(folder_path) if not item.startswith('._')]
    filtered_filenames = sorted(filtered_filenames)
    for filename in filtered_filenames:
        file_path = os.path.join(folder_path, filename)
        curve = load_float_curve_from_txt_file(file_path)
        curves.append(curve)
        filenames.append(filename)
    
    return curves, filenames



def load_pixelated_curve_from_txt_file(file_path):
    """
    Load a curve from a txt file
    """
    points = np.loadtxt(file_path, dtype=int)
    unique_rows, idx = np.unique(points, axis=0, return_index=True)
    pixelated_curve = unique_rows[np.argsort(idx)]
    return pixelated_curve


def load_pixelated_curves_from_txt_files(folder_path):
    """
    Load all curves from a folder containing txt files
    """
    curves = []
    filenames = []
    filtered_filenames = [item for item in os.listdir(folder_path) if not item.startswith('._')]
    filtered_filenames = sorted(filtered_filenames)
    for filename in filtered_filenames:
        file_path = os.path.join(folder_path, filename)
        curve = load_pixelated_curve_from_txt_file(file_path)
        curves.append(curve)
        filenames.append(filename)
    
    return curves, filenames



def load_pixelated_curve_from_image(image_path, start_point):
    """
    Load a curve containded ina a binary image
    """
    # get curve from image
    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    treepath = imscc.build_tree(o_image, start_point)                                           

    k = 2
    branch = []
    curve_elem = treepath[k]
    while type(curve_elem) is tuple:
        branch.append(curve_elem)
        curve_elem = treepath[k]
        k += 1
    
    bx = np.array([point[1] for point in branch])
    by = np.array([point[0] for point in branch])
    pixel_curve = np.column_stack([bx, by])
    return pixel_curve



def plot_two_curves(curve1, curve2, plot_title="Curves Comparison", label1='Curve 1', label2='Curve 2'):
    """
    Plot the results of the polynomial fitting
    """
    plt.figure(figsize=(18, 12))
    plt.plot(curve1[:, 0], curve1[:, 1], 'o-', color="darkturquoise", alpha=0.6, markersize=2, linewidth=1, label=label1)
    plt.plot(curve2[:, 0], curve2[:, 1], 'o-', color="crimson", alpha=0.6, markersize=2, linewidth=1, label=label2)


    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()



def display_curve_on_image(image_path, curve):

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError("Could not load image")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    x = curve[:, 0]
    y = curve[:, 1]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.plot(x, y, linewidth=2, color='red', alpha=0.5)   # curve overlay
    plt.scatter(x, y, s=5, color='yellow', alpha=0.5)     # optional: show sample points
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def display_curve_on_image_2(image_path, curve1, curve2):
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError("Could not load image")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    x = curve1[:, 0]
    y = curve1[:, 1]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.plot(curve2[:, 0], curve2[:, 1], 'o-', color="lawngreen",
     alpha=0.5, markersize=2, linewidth=1, label="pixel")
    plt.plot(x, y, linewidth=2, color='red', alpha=0.5)   # curve overlay
    plt.scatter(x, y, s=5, color='yellow', alpha=0.5)     # optional: show sample points
    

    plt.axis('off')
    plt.tight_layout()
    plt.show()