
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import time
import yaml



from scipy.interpolate import splprep, splev, UnivariateSpline, CubicSpline, make_interp_spline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

import Curve_utils as cutils
import utils.Smoothing as smooth
import ImageToSCC as imscc
import Morphology_Measurements_Single_Curve as measure

import utils.load_and_write as lw
from SCC_Tree_old import SCC_Tree
from Morphology_Measures import Morphology_Measures
from Tortuosity_Measures import TortuosityMeasures



matplotlib.use('Qt5Agg')

def build_local_interpolated_tree(tree_path):
    """

    :param tree_path:
    :return:
    """
    p1 = 2
    p2 = 0
    branch = []
    branches = {}
    interp_branches = {}
    interp_tree = [2, 1, tree_path[2]]

    len_vec = []
    for k in range(2, len(tree_path)):
        data = tree_path[k]

        if type(data) is not tuple:
            # print(branch[-1])
            if data != 1:
                if not p1:
                    p1 = data
                else:
                    p2 = data
            if p1 and p2:
                if p1 < p2:
                    bx = np.array([point[0] for point in branch])
                    by = np.array([point[1] for point in branch])
                    size_vec = round(len(bx) * 0.25)
                    len_vec.append(size_vec)
                    # D = interp_curve(round(len(bx) * 0.25), by, bx)

                    # pixel_curve = interp_curve(int(len(bx) * 0.25), bx, by)
                    # pixel_curve = interp_curve(len(bx), bx, by)
                                        
                    # Apply smoothing
                    pixel_curve = np.column_stack([bx, by])
                    

                    # smoothed_curve, tck = smooth_with_splprep(pixel_curve, smoothing_factor=8.0)
                    
                    # smoothed_curve = smooth_with_univariate_spline(pixel_curve, smoothing_factor=15.0)
                    
                    # smoothed_curve, cs = smooth_with_cubic_spline(pixel_curve)

                    # smoothed_curve = smooth_with_arc_length(bx, by, 25)
                    # smoothed_curve = smooth_Savitzky_Golay(bx, by,size_vec, 11, 5)

                    smoothed_curve = smooth.smooth_with_regularization(pixel_curve, 0.5, 0.5)
                    
                    smoothed_curve[0] = np.array([bx[0], by[0]])
                    smoothed_curve[-1] = np.array([bx[-1], by[-1]])

                    plot_results(pixel_curve, smoothed_curve)

                    
                    D = smoothed_curve

                    plot_results(smoothed_curve, pixel_curve)

                    interp_branch = [(point[0], point[1]) for point in D]
                    branches[(p1, p2)] = branch
                    interp_branches[(p1, p2)] = interp_branch

                    for i in range(1, len(interp_branch)):
                        interp_tree.append(interp_branch[i])
                else:
                    interp_branch = interp_branches[(p2, p1)]
                    interp_branch.reverse()
                    for i in range(1, len(interp_branch)):
                        interp_tree.append(interp_branch[i])

                branch = [branch[-1]]
                p1 = p2
                p2 = 0
            interp_tree.append(data)
        else:
            branch.append(data)

    print("Points on branches: {}".format(len_vec))
    return interp_tree



def load_tort_file(filename):
    values = []
    with open(filename, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.split(",") if p.strip()]  # split by comma, drop blanks
            if len(parts) >= 2:
                first_val = float(parts[1])  # first number after filename
                values.append(first_val)

    return values



def test_curve_interpolation(path, image_folder, des_file):
    real_tort = load_tort_file(os.path.join(path, "measurements_curves.csv"))

    pattern = re.compile(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)')
    pattern_num = re.compile(r"_(\d+)_X(\d+)")
    file_data = []

    torts = []
    dists = []
    with open(os.path.join(path, des_file), 'r', encoding='utf-8') as f:        
        im_num = 0  
        row = 0
        diffs_tort = np.zeros((50, 3))
        all_dists = np.zeros((50, 3))
        for line in f:
            new_row = False
            match = pattern.search(line)               

            if match:
                fname = match.group(1)

                match2 = re.search(r'_(\d+)_X(\d+)', fname)
                if match2:
                    file_num = int(match2.group(1))
                    scale = float(match2.group(2)) / 10.
                
                # reset metrics counter
                if im_num != file_num:                    
                    im_num = file_num
                    new_row = True


                x = float(match.group(2))
                y = float(match.group(3))
                file_data.append({'filename': fname, 'x': x, 'y': y})
                sp = (int(y), int(x))

                o_image = cv2.imread(os.path.join(path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)           
                
                # imscc.display_tree(scc_tree, dist)
                # [X, Y] = imscc.plot_tree(scc_tree, dist)
                
                branch = []

                k = 2
                curve_elem = treepath[k]
                while type(curve_elem) is tuple:
                    branch.append(curve_elem)
                    curve_elem = treepath[k]
                    k += 1
                
                bx = np.array([point[0] for point in branch])
                by = np.array([point[1] for point in branch])

                if new_row:
                    size_vec = round(len(bx) * 1)   
                    svec = round(len(bx) * 0.50) 
                                                
                                                           
                pixel_curve = np.column_stack([bx, by])

                # smoothed_curve = pixel_curve
                # smoothed_curve = smooth.smooth_Savitzky_Golay(bx, by, len(bx), 3, 1) # best -> 7, 1
                # smoothed_curve = smooth.smooth_with_regularization(pixel_curve, 0.03) # best -> 0.054 or 0.057
                # smoothed_curve = smooth.smooth_with_univariate_spline(pixel_curve, smoothing_factor=0.055, num_points=size_vec)       # best -> 0.057       

                pixel_curve = smooth.uniform_resample(pixel_curve, svec)
                smoothed_curve = smooth.gaussian_smooth(pixel_curve, sigma=0.0, beta=0.2)
        
                ys = smoothed_curve[:, 0]
                xs = smoothed_curve[:, 1]

                param_curve = np.column_stack([xs, ys])
                # s_eq, x_eq, y_eq, _ = smooth.arclength_parametrization(xs, ys, n_samples=size_vec, method="linear")
                # param_curve = np.column_stack([x_eq, y_eq])


                # plot_results(pixel_curve, pixel_curve)
                

                [T, T_n] = measure.SCC(param_curve)      
                dm = measure.DM(param_curve)     
                L = measure.ArcLen(param_curve)

                if new_row and len(torts) > 1:
                    rt = real_tort[row]
                    diffs_tort[row,:] = abs(np.array(torts) - rt)
                    all_dists[row,:] = dists
                    torts = []
                    dists = []
                    row += 1 

                torts.append(T)     

                 # compute the distance between curves                 
                name, _ = os.path.splitext(fname)                
                original_curve = read_coordinates(os.path.join(path, "points", name + ".txt")) * scale
                
                D = smooth.average_min_distance(param_curve, original_curve)
                dists.append(D)       

                # plot_three_curves(original_curve, param_curve, pixel_curve[:, [1, 0]], labels=["Original", "Smoothed", "Pixelated"])                                
                
        rt = real_tort[row]
        diffs_tort[row,:] = abs(np.array(torts) - rt)
        all_dists[row,:] = dists
        for row, r in enumerate(diffs_tort):
            print('{:<4}{} - {}'.format(row, np.array2string(r, precision=4), np.array2string(all_dists[row, :], precision=4)))
        print('Tort - {} - all: {:.5f} - Dist {} - all: {:.5f}'.format(np.array2string(np.mean(diffs_tort, axis=0), suppress_small=True, precision=4), np.mean(diffs_tort), 
                                                                       np.array2string(np.mean(all_dists, axis=0), suppress_small=True, precision=4), np.mean(all_dists)))



def test_curve_smoothing(path, image_folder, des_file, rate=0.25):
    real_tort = load_tort_file(os.path.join(path, "measurements_curves.csv"))

    pattern = re.compile(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)')

    torts = []
    torts_o = []
    dists = []
    variation = []
    with open(os.path.join(path, des_file), 'r', encoding='utf-8') as f:        
        im_num = 0  
        row = 0
        diffs_tort = np.zeros((50, 3))
        all_dists = np.zeros((50, 3))
        for line in f:
            new_row = False
            match = pattern.search(line)               

            if match:
                fname = match.group(1)

                match2 = re.search(r'_(\d+)_X(\d+)', fname)
                if match2:
                    file_num = int(match2.group(1))
                    scale = float(match2.group(2)) / 10.
                

                x = float(match.group(2))
                y = float(match.group(3))
                sp = (int(y), int(x))

                o_image = cv2.imread(os.path.join(path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)           
                
                name, _ = os.path.splitext(fname)                
                original_curve = read_coordinates(os.path.join(path, "points", name + ".txt")) * scale
                branch = []

                k = 2
                branch = []                
                while type(treepath[k]) is tuple:
                    branch.append(treepath[k])                    
                    k += 1
                
                bx = np.array([point[0] for point in branch])
                by = np.array([point[1] for point in branch])

                 # reset metrics counter
                if im_num != file_num:                    
                    im_num = file_num
                    new_row = True
                else:
                    new_row = False

                if new_row:
                    size_vec = round(len(bx) * rate)   
                                                
                pixel_curve = np.column_stack([bx, by])        

                s_eq, x_eq, y_eq, _ = smooth.arclength_parametrization(bx, by, n_samples=size_vec, method="linear")
                pixel_param_curve = np.column_stack([x_eq, y_eq])

                smoothed_curve = pixel_param_curve
                # smoothed_curve = smooth_Savitzky_Golay(x_eq, y_eq, len(x_eq), 11, 3)
                # smoothed_curve = smooth.smooth_with_regularization(pixel_param_curve, 0.057)
                # smoothed_curve = smooth_with_univariate_spline(pixel_param_curve, smoothing_factor=0.047, num_points=size_vec)

                
                
                # plot_results(pixel_curve[:, [1, 0]], smoothed_curve[:, [1, 0]])

                

                if new_row and len(torts) > 1:                    
                    rt = real_tort[row]
                    diffs_tort[row,:] = np.abs(np.array(torts) - np.array(torts_o))
                    all_dists[row,:] = dists
                    variation.append(np.std(np.array(torts)) / np.mean(np.array(torts)))
                    print('{:<4} {:<20} {:.4f}  {}  {:.4f} - {} - {}'.format(row, fname, To,
                                                              np.array2string(np.array(torts), precision=4), variation[-1],
                                                              np.array2string(diffs_tort[row,:], precision=4), 
                                                              np.array2string(np.array(dists), precision=4)))
                    torts = []
                    torts_o = []
                    dists = []
                    row += 1 
                
                [T, T_n] = measure.SCC(smoothed_curve)
                [To, _] = measure.SCC(original_curve)

                dm = measure.DM(smoothed_curve)     
                L = measure.ArcLen(smoothed_curve)                
                D = (smooth.average_min_distance(smoothed_curve[:, [1, 0]], original_curve) / scale) / smoothed_curve.shape[0]

                torts.append(T)
                torts_o.append(To)
                dists.append(D)  

                     

                # plot_three_curves(original_curve, param_curve, pixel_curve[:, [1, 0]], labels=["Original", "Smoothed", "Reference"])
                # plot_three_curves(original_curve, pixel_curve_2[:, [1, 0]], pixel_curve[:, [1, 0]], labels=["Original", "Smoothed", "Reference"])
                
                
        rt = real_tort[row]
        diffs_tort[row,:] = np.array(torts) - rt 
        all_dists[row,:] = dists
        # for row, r in enumerate(diffs_tort):
            # print('{:<4}{} - {}'.format(row, np.array2string(r, precision=4), np.array2string(all_dists[row, :], precision=4)))
        print('Tort - {} - all: {:.4f}, {:.4f} - Dist {} - all: {:.4f}'.format(np.array2string(np.mean(diffs_tort, axis=0), 
                                                                        suppress_small=True, precision=4), np.mean(diffs_tort), np.mean(np.array(variation)),
                                                                        np.array2string(np.mean(all_dists, axis=0), suppress_small=True, precision=4), np.mean(all_dists)))




def test_curve_smoothing_all(path, image_folder, des_file, rate=0.25):
    torts = []
    torts_o = []
    dists = []
    variation = []

    with open(os.path.join(path, des_file), 'r', encoding='utf-8') as f:        
        im_num = 0  
        row = 0
        diffs_tort = np.zeros((50, 3))
        all_dists = np.zeros((50, 3))
        for idx, line in enumerate(f):            
            match1 = re.search(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', line)               
            if match1:
                fname = match1.group(1)

                match2 = re.search(r'_(\d+)_X(\d+)', fname)
                if match2:
                    scale = float(match2.group(2)) / 10.
                

                x = float(match1.group(2))
                y = float(match1.group(3))
                sp = (int(y), int(x))

                o_image = cv2.imread(os.path.join(path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)           
                
                name, _ = os.path.splitext(fname)                
                original_curve = read_coordinates(os.path.join(path, "points", name + ".txt")) * scale
                branch = []

                k = 2
                branch = []                
                while type(treepath[k]) is tuple:
                    branch.append(treepath[k])                    
                    k += 1
                
                bx = np.array([point[0] for point in branch])
                by = np.array([point[1] for point in branch])
                size_vec = round(len(bx) * rate)   
                                                
                # pixel_curve = np.column_stack([bx, by])
        
                s_eq, x_eq, y_eq, _ = smooth.arclength_parametrization(bx, by, n_samples=size_vec, method="linear")
                pixel_param_curve = np.column_stack([x_eq, y_eq])

                smoothed_curve = smooth.smooth_with_regularization(pixel_param_curve, 0.057)
                [To, _] = measure.SCC(original_curve)
                [T, _] = measure.SCC(smoothed_curve)
        
                Td = abs(To - T)
                D = (smooth.average_min_distance(smoothed_curve[:, [1, 0]], original_curve) / scale) / smoothed_curve.shape[0]

                print('{:<4} {:<20} {:.4f}  {:.4f}  {:.4f}'.format(idx, fname, To, Td, D))








    













def smooth_with_splprep(pixel_curve, smoothing_factor=0.5):
    """
    Smooth using parametric B-spline (handles multi-valued functions)
    """

    size = int(len(pixel_curve[:, 0]) * 0.25)
    # Extract coordinates
    x = pixel_curve[:, 0]
    y = pixel_curve[:, 1]
    
    # Create parametric representation
    t = np.arange(len(x))
    
    # Fit parametric B-spline
    tck, u = splprep([x, y], s=smoothing_factor * len(x), per=0)
    
    # Generate smooth curve
    u_new = np.linspace(0, 1, size)
    x_smooth, y_smooth = splev(u_new, tck)
    
    return np.column_stack([x_smooth, y_smooth]), tck







def smooth_with_cubic_spline(pixel_curve):
    """
    Smooth using CubicSpline interpolation
    """

    size = int(len(pixel_curve[:, 0]) * 0.25)

    # Sort by x-coordinate
    sorted_indices = np.argsort(pixel_curve[:, 0])
    x_sorted = pixel_curve[sorted_indices, 0]
    y_sorted = pixel_curve[sorted_indices, 1]
    
    # Remove duplicates
    unique_mask = np.r_[True, ~np.isclose(np.diff(x_sorted), 0)]
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    # Fit cubic spline
    cs = CubicSpline(x_unique, y_unique)
    
    # Generate smooth curve
    x_smooth = np.linspace(x_unique.min(), x_unique.max(), size)
    y_smooth = cs(x_smooth)
    
    return np.column_stack([x_smooth, y_smooth]), cs






def process_pixel_curve(pixel_array, degree=3, return_metrics=True):
    """
    Process contiguous pixel curve with numpy.polyfit
    
    Parameters:
    pixel_array: N×2 array of [x,y] coordinates
    degree: polynomial degree for fitting
    return_metrics: whether to return quality metrics
    
    Returns:
    coefficients, polynomial function, and optional metrics
    """
    # Sort by x-coordinate to ensure contiguous order
    sorted_indices = np.argsort(pixel_array[:, 0])
    sorted_pixels = pixel_array[sorted_indices]
    
    x = sorted_pixels[:, 0]
    y = sorted_pixels[:, 1]
    
    # Apply numpy.polyfit
    coefficients = np.polyfit(x, y, deg=degree)
    poly_func = np.poly1d(coefficients)
    
    if return_metrics:
        # Calculate quality metrics
        y_pred = poly_func(x)
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean squared error
        mse = np.mean((y - y_pred) ** 2)
        
        # Maximum error
        max_error = np.max(np.abs(y - y_pred))
        
        metrics = {
            'r_squared': r_squared,
            'mse': mse,
            'max_error': max_error,
            'n_points': len(x)
        }
        
        return coefficients, poly_func, metrics
    else:
        return coefficients, poly_func
    






def read_coordinates(file_path):
    """
    Reads a text file containing comma-separated coordinates (x, y)
    and returns a NumPy array of shape (n, 2).
    
    Parameters:
        file_path (str): Path to the text file.
    
    Returns:
        np.ndarray: Array containing the (x, y) coordinates.
    """
    # Load the file assuming each line has "x,y"
    coordinates = np.loadtxt(file_path, delimiter=',')
    return coordinates




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




def plot_three_curves(curve1, curve2, curve3, labels=None, title="Three curves"):
    """
    Plots three curves, each given as an array of shape (n, 2).

    Parameters:
        curve1, curve2, curve3 (np.ndarray): Arrays with shape (n, 2) = (x, y).
        labels (list or tuple, optional): Names for the three curves.
        title (str): Plot title.
    """
    if labels is None:
        labels = ["Curve 1", "Curve 2", "Curve 3"]

    plt.figure(figsize=(12, 12))

    plt.plot(curve1[:, 0], curve1[:, 1], '-o', markersize=2, label=labels[0])
    plt.plot(curve2[:, 0], curve2[:, 1], '-o', markersize=2.5, label=labels[1])
    plt.plot(curve3[:, 0], curve3[:, 1], '-o', markersize=2, label=labels[2])

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()



def measure_neuron_tree(config_file, image_filename):
    with open(config_file, 'r') as conf_file:
        config_data = yaml.safe_load(conf_file)
    im = None
    images = config_data['skeleton_images']
    for i, item in enumerate(images):
        if item['filename'] == image_filename:
            im = config_data['skeleton_images'][i]
            break

    im = config_data['skeleton_images'][i]
    if im['subfolder'] is not None:
        image_path = os.path.join(config_data["image_folder"], im['subfolder'], im['filename'])
    else:
        image_path = os.path.join(config_data["image_folder"], im['filename'])

    assert os.path.isfile(image_path), "The image {} doesn't exist".format(image_path)

    start_position = im['start_position'][0]
    sp = (start_position['position']['y'],start_position['position']['x'])

    o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    treepath = imscc.build_tree(o_image, sp)
    interp_tree = build_local_interpolated_tree(treepath)

    [scc_tree, dist] = imscc.build_scc_tree(interp_tree)

    # imscc.display_tree(scc_tree, dist)
    [X, Y] = imscc.plot_tree(scc_tree, dist)

    [m_angle, t_angles] = Morphology_Measures.tree_scc_branch_anlge_2(scc_tree)
    [T, T_n] = TortuosityMeasures.Tree_SCC(scc_tree)      
    [dm_m, dm_st, dm_t] = TortuosityMeasures.DM_Tree(interp_tree)      
    L = Morphology_Measures.tree_length(interp_tree)
    
    [seg, bifur, term] = Morphology_Measures.tree_scc_count_features(scc_tree)
    [branch_mean_length, branch_sum_length, branch_lengths] = Morphology_Measures.tree_branch_length(interp_tree)
    T_c = Morphology_Measures.tree_scc_circularity(scc_tree)
    T_l = Morphology_Measures.tree_scc_linearity(scc_tree)
    [C, C_m] = Morphology_Measures.convex_concav(scc_tree)

    print("{} \tTort = {} \tTort_norm = {} \tAngle = {}".format(im['filename'], T, T_n, m_angle))  

    return interp_tree    



def test_load_and_write():
    # lw.save_pixelated_curve_set('/Volumes/HOUSE MINI/IMAGENES/curves_500_5', 'coordinates_curves.txt', 'images',
    #                             '/Volumes/HOUSE MINI/IMAGENES/curves_500_5/pixel_curves') 
    # lw.load_curves_from_txt_file('/Volumes/HOUSE MINI/IMAGENES/curves_500_5/pixel_curves')

    # lw.save_pixelated_curve_set('/Users/zianfanti/IIMAS/images_databases/curves_500_5', 'coordinates_curves.txt', 'images',
    #                             '/Users/zianfanti/IIMAS/images_databases/curves_500_5/pixel_curves') 
    
    [c, filename] = lw.load_pixelated_curves_from_txt_files('/Users/zianfanti/IIMAS/images_databases/curves_500_5/pixel_curves')
    [cf, filename] = lw.load_float_curves_from_txt_files('/Users/zianfanti/IIMAS/images_databases/curves_500_5/target_scaled')

    for idx, (c1, c2) in enumerate(zip(c,cf)):
        lw.plot_two_curves(c2,c1, plot_title=filename[idx])



# measure_neuron_tree('neuron_config_2.yaml', 'fish01_2.CNG.tif')

# test_curve_interpolation("/Users/zianfanti/IIMAS/images_databases/curves", "images", "coordinates_curves.txt")


start_time = time.perf_counter()

# test_curve_interpolation("/Users/zianfanti/IIMAS/images_databases/curves", "images", "coordinates_curves.txt")
# test_curve_smoothing('/Users/zianfanti/IIMAS/images_databases/curves', "images", "coordinates_curves.txt", rate=0.50)
# test_curve_smoothing('/Volumes/HOUSE MINI/IMAGENES/curves', "images", "coordinates_curves.txt", rate=0.25)
# test_curve_smoothing_all('/Volumes/HOUSE MINI/IMAGENES/curves_200_5_1', "images", "coordinates_curves.txt", rate=0.25)


# for r in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
#     print("---- Rate: {} ----".format(r))
#     test_curve_smoothing('/Users/zianfanti/IIMAS/images_databases/curves', "images", "coordinates_curves.txt", rate=r)

test_load_and_write()



end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds")

