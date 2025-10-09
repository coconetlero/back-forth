import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml

from scipy.interpolate import splprep, splev, UnivariateSpline, CubicSpline
from scipy.interpolate import make_interp_spline

import ImageToSCC as imscc
from SCC_Tree import SCC_Tree
from Morphology_Measures import Morphology_Measures
from Tortuosity_Measures import TortuosityMeasures

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
                    bx = np.array([point[1] for point in branch])
                    by = np.array([point[0] for point in branch])
                    # D = interp_curve(round(len(bx) * 0.25), by, bx)

                    # Apply smoothing
                    pixel_curve = np.column_stack([bx, by])


                    # smoothed_curve, tck = smooth_with_splprep(pixel_curve, smoothing_factor=5.0)
                    # smoothed_curve, spline_uni = smooth_with_univariate_spline(pixel_curve, smoothing_factor=12.0)
                    
                    smoothed_curve, cs = smooth_with_cubic_spline(pixel_curve)

                    plt.figure(figsize=(12, 6))
                    plt.plot(pixel_curve[:, 0], pixel_curve[:, 1], 'bo-', alpha=0.7, 
                            markersize=6, label='Original Pixel Curve')
                    plt.plot(smoothed_curve[:, 0], smoothed_curve[:, 1], 'purple', linewidth=2, 
                            label='Cubic Spline')
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                    plt.title('Curve Smoothing with Cubic Spline')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.show()

                    smoothed_curve[0] = np.array([bx[0], by[0]])
                    smoothed_curve[-1] = np.array([bx[-1], by[-1]])
                    D = smoothed_curve

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

    return interp_tree


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

    print("{} \tTort = {} \tAngle = {}".format(im['filename'], T, m_angle))  

    return interp_tree    



def smooth_with_splprep(pixel_curve, smoothing_factor=0.5):
    """
    Smooth using parametric B-spline (handles multi-valued functions)
    """
    # Extract coordinates
    x = pixel_curve[:, 0]
    y = pixel_curve[:, 1]
    
    # Create parametric representation
    t = np.arange(len(x))
    
    # Fit parametric B-spline
    tck, u = splprep([x, y], s=smoothing_factor * len(x), per=0)
    
    # Generate smooth curve
    u_new = np.linspace(0, 1, 200)
    x_smooth, y_smooth = splev(u_new, tck)
    
    return np.column_stack([x_smooth, y_smooth]), tck



def smooth_with_univariate_spline(pixel_curve, smoothing_factor=None):
    """
    Smooth using UnivariateSpline (works when y is function of x)
    """
    # Sort by x-coordinate
    sorted_indices = np.argsort(pixel_curve[:, 0])
    x_sorted = pixel_curve[sorted_indices, 0]
    y_sorted = pixel_curve[sorted_indices, 1]
    
    # Remove duplicates for spline fitting
    unique_mask = np.r_[True, ~np.isclose(np.diff(x_sorted), 0)]
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    if smoothing_factor is None:
        smoothing_factor = len(x_unique)
    
    # Fit spline
    spline = UnivariateSpline(x_unique, y_unique, s=smoothing_factor)
    
    # Generate smooth curve
    x_smooth = np.linspace(x_unique.min(), x_unique.max(), int(len(pixel_curve[:, 0]) * 0.25))
    y_smooth = spline(x_smooth)
    
    return np.column_stack([x_smooth, y_smooth]), spline



def smooth_with_cubic_spline(pixel_curve):
    """
    Smooth using CubicSpline interpolation
    """
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
    x_smooth = np.linspace(x_unique.min(), x_unique.max(), 200)
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
    



def plot_results(image_pixels, poly_func, metrics):
    """
    Plot the results of the polynomial fitting
    """
    plt.figure(figsize=(12, 6))

    # Plot pixel data with stair-step appearance
    plt.step(image_pixels[:, 0], image_pixels[:, 1], 's-', 
             where='mid', color='blue', linewidth=1, markersize=6,
             label='Pixel Coordinates', alpha=0.7)

    # Plot fitted smooth curve
    x_smooth = np.linspace(image_pixels[:, 0].min(), image_pixels[:, 0].max(), 100)
    y_smooth = poly_func(x_smooth)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
             label=f'Fitted Polynomial (deg=2, R²={metrics["r_squared"]:.3f})')

    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.title('Polynomial Fit to Pixelated Image Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



measure_neuron_tree('neuron_config_2.yaml', 'ex_02.tif')


