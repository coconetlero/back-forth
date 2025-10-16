import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml

from scipy.interpolate import splprep, splev, UnivariateSpline, CubicSpline, make_interp_spline
from scipy.signal import savgol_filter

import ImageToSCC as imscc
from SCC_Tree import SCC_Tree
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

                    smoothed_curve = smooth_with_arc_length(bx, by, 25)
                    # smoothed_curve = smooth_Savitzky_Golay(bx, by,size_vec, 11, 5)

                    smoothed_curve = smooth_with_regularization(pixel_curve, 0.5, 0.5)
                    
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





def interp_curve(num_points, px, py):
    if num_points > 1:
        # equally spaced in arclength
        N = np.transpose(np.linspace(0, 1, num_points))

        # how many points will be uniformly interpolated?
        nt = N.size

        # number of points on the curve
        n = px.size
        pxy = np.column_stack([px, py])
        pt = np.zeros((nt, 2))

        # Compute the arclength of each segment.
        chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
        # Normalize the arclengths to a unit total
        chordlen = chordlen / np.sum(chordlen)
        # cumulative arclength
        cumarc = np.append(0, np.cumsum(chordlen))

        tbins = np.digitize(N, cumarc)  # bin index in which each N is in

        # catch any problems at the ends
        tbins[np.where(np.bitwise_or(tbins <= 0, (N <= 0)))] = 1
        tbins[np.where(np.bitwise_or(tbins >= n, (N >= 1)))] = n - 1

        s = np.divide((N - cumarc[tbins - 1]), chordlen[tbins - 1])
        pt = pxy[tbins - 1, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

        pt[0] = np.array([px[0], py[0]])
        pt[-1] = np.array([px[-1], py[-1]])
        return pt
    else:
        return np.array([(px[0], py[0]),(px[-1], py[-1])])
    



def smooth_with_arc_length(x, y, size):
    # Calculate cumulative arc length
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    length = np.sum(ds)
    ratio = abs(1 - (length / len(dx))) if (length / len(dx)) > 1 else (length / len(dx)) 
    ratio1 = len(dx) / length 
    print("Curve ratio: {}".format(ratio))  

    size = round(len(dx) * (ratio)) 
    s = np.concatenate(([0], np.cumsum(ds)))
    
    # Resample to uniform arc length
    s_uniform = np.linspace(0, s[-1], size)
    x_uniform = np.interp(s_uniform, s, x)
    y_uniform = np.interp(s_uniform, s, y)
    
    smooth_curve = np.column_stack([x_uniform, y_uniform])
    return smooth_curve





def smooth_Savitzky_Golay(x, y, size, window_length=11, polyorder=3):
    # Calculate cumulative arc length
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    # Resample to uniform arc length
    s_uniform = np.linspace(0, s[-1], size)
    x_uniform = np.interp(s_uniform, s, x)
    y_uniform = np.interp(s_uniform, s, y)
    
    # Apply S-G filter
    x_smooth = savgol_filter(x_uniform, window_length, polyorder)
    y_smooth = savgol_filter(y_uniform, window_length, polyorder)
    
    smooth_curve = np.column_stack([x_smooth, y_smooth])
    return smooth_curve



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



def smooth_with_univariate_spline(pixel_curve, smoothing_factor=None):
    """
    Smooth x and y separately as functions of arc length
    """

    size = int(len(pixel_curve[:, 0]) * 0.25)

    # Calculate arc length parameter
    dx = np.diff(pixel_curve[:, 0])
    dy = np.diff(pixel_curve[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))  # Arc length parameter
    
    x = pixel_curve[:, 0]
    y = pixel_curve[:, 1]
    
    # Smooth x and y separately as functions of arc length
    spline_x = UnivariateSpline(s, x, s=smoothing_factor * len(x))
    spline_y = UnivariateSpline(s, y, s=smoothing_factor * len(y))
    
    # Generate smooth arc length parameter
    s_smooth = np.linspace(0, s.max(), size)
    
    # Get smoothed x and y
    x_smooth = spline_x(s_smooth)
    y_smooth = spline_y(s_smooth)
    
    return np.column_stack([x_smooth, y_smooth])



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



def smooth_with_regularization(pixel_curve, alpha=0.1, beta=0.1):
    """
    Smooth both x and y with curvature regularization
    """
    size = int(len(pixel_curve[:, 0]) * 0.25)
    
    # Calculate arc length
    dx = np.diff(pixel_curve[:, 0])
    dy = np.diff(pixel_curve[:, 1])
    s = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy**2))))
    
    x_orig = pixel_curve[:, 0]
    y_orig = pixel_curve[:, 1]
    
    # Fit smoothing splines to both coordinates
    # Using different smoothing factors for demonstration
    spline_x = UnivariateSpline(s, x_orig, s=alpha * len(s))
    spline_y = UnivariateSpline(s, y_orig, s=beta * len(s))
    
    # Generate smooth curve
    s_smooth = np.linspace(s[0], s[-1], size)
    x_smooth = spline_x(s_smooth)
    y_smooth = spline_y(s_smooth)
    
    return np.column_stack([x_smooth, y_smooth])


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
    


def plot_results(pixel_curve, smoothed_curve):
    """
    Plot the results of the polynomial fitting
    """
    plt.figure(figsize=(12, 12))
    plt.plot(pixel_curve[:, 0], pixel_curve[:, 1], 'bo-', alpha=0.4, 
            markersize=6, label='Original Pixel Curve')
    plt.plot(smoothed_curve[:, 0], smoothed_curve[:, 1], 'ro-', linewidth=2, 
            label='Cubic Spline')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Curve Smoothing with Cubic Spline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
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


measure_neuron_tree('neuron_config_2.yaml', 'fish01_2.CNG.tif')


