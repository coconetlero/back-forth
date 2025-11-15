import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import time
import yaml

from typing import Tuple, Optional

from scipy.interpolate import splprep, splev, UnivariateSpline, CubicSpline, make_interp_spline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

import ImageToSCC as imscc
import Morphology_Measurements_Single_Curve as measure

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

                    # smoothed_curve = smooth_with_arc_length(bx, by, 25)
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



def test_curve_interpolation(path, image_folder, des_file, rate=0.25):
    real_tort = load_tort_file(os.path.join(path, "measurements_curves.csv"))

    pattern = re.compile(r'(\S+)\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)')
    pattern_num = re.compile(r"_(\d+)_X(\d+)")
    file_data = []

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
                file_data.append({'filename': fname, 'x': x, 'y': y})
                sp = (int(y), int(x))

                o_image = cv2.imread(os.path.join(path, image_folder, fname), cv2.IMREAD_GRAYSCALE)
                treepath = imscc.build_tree(o_image, sp)           
                
                name, _ = os.path.splitext(fname)                
                original_curve = read_coordinates(os.path.join(path, "points", name + ".txt")) * scale
                branch = []

                k = 2
                curve_elem = treepath[k]
                while type(curve_elem) is tuple:
                    branch.append(curve_elem)
                    curve_elem = treepath[k]
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
                

                # smoothed_curve = pixel_curve
                # smoothed_curve = smooth_Savitzky_Golay(bx, by, len(bx), 9, 1) # best -> 9, 1
                # smoothed_curve = smooth_with_regularization(pixel_curve, 0.055) # best -> 0.054 or 0.057
                # smoothed_curve = smooth_with_univariate_spline(pixel_curve, smoothing_factor=0.05, num_points=size_vec)       # best -> 0.057         
        

                s_eq, x_eq, y_eq, _ = arclength_param(bx, by, n_samples=size_vec, method="linear")
                pixel_param_curve = np.column_stack([x_eq, y_eq])

                # smoothed_curve = smooth_Savitzky_Golay(x_eq, y_eq, len(x_eq), 11, 3)
                smoothed_curve = smooth_with_regularization(pixel_param_curve, 0.055)
                # smoothed_curve = smooth_with_univariate_spline(pixel_param_curve, smoothing_factor=0.047, num_points=size_vec)


                xs = smoothed_curve[:, 0]
                ys = smoothed_curve[:, 1]
                param_curve = np.column_stack([xs, ys])

                # param_curve = pixel_param_curve

                # s_eq, x_eq, y_eq, _ = arclength_param(xs, ys, n_samples=size_vec, method="linear")
                # param_curve = np.column_stack([x_eq, y_eq])

                
                plot_results(pixel_curve[:, [1, 0]], param_curve[:, [1, 0]])


                if new_row and len(torts) > 1:                    
                    rt = real_tort[row]
                    diffs_tort[row,:] = np.abs(np.array(torts) - np.array(torts_o))
                    all_dists[row,:] = dists
                    variation.append(np.std(np.array(torts)) / np.mean(np.array(torts)))
                    print('{:<4} {:.4f}  {}  {:.4f} - {} - {}'.format(row, To,
                                                              np.array2string(np.array(torts), precision=4), variation[-1],
                                                              np.array2string(diffs_tort[row,:], precision=4), 
                                                              np.array2string(np.array(dists), precision=4)))
                    torts = []
                    torts_o = []
                    dists = []
                    row += 1 
                

                [T, T_n] = measure.SCC(param_curve)
                [To, _] = measure.SCC(original_curve)

                dm = measure.DM(param_curve)     
                L = measure.ArcLen(param_curve)                
                D = average_min_distance(param_curve[:, [1, 0]], original_curve)

                torts.append(T)
                torts_o.append(To)
                dists.append(D)  

                     

                # plot_three_curves(original_curve, param_curve, pixel_curve[:, [1, 0]], labels=["Original", "Smoothed", "Reference"])
                # plot_three_curves(original_curve, pixel_param_curve[:, [1, 0]], pixel_curve[:, [1, 0]], labels=["Original", "Smoothed", "Reference"])
                
                
        rt = real_tort[row]
        diffs_tort[row,:] = np.array(torts) - rt 
        all_dists[row,:] = dists
        # for row, r in enumerate(diffs_tort):
            # print('{:<4}{} - {}'.format(row, np.array2string(r, precision=4), np.array2string(all_dists[row, :], precision=4)))
        print('Tort - {} - all: {:.4f}, {:.4f} - Dist {} - all: {:.4f}'.format(np.array2string(np.mean(diffs_tort, axis=0), 
                                                                                           suppress_small=True, precision=4), np.mean(diffs_tort), np.mean(np.array(variation)),
                                                                       np.array2string(np.mean(all_dists, axis=0), suppress_small=True, precision=4), np.mean(all_dists)))


       



def arclength_param(
    x, y,
    spacing: Optional[float] = None,
    n_samples: Optional[int] = 200,
    method: str = "linear",
    smooth_sigma: float = 0.0,
    closed: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample a 2D polyline (x,y) onto a uniform arc-length parameter.

    Parameters
    ----------
    x, y : array-like
        Ordered coordinates along the curve.
    spacing : float, optional
        Desired step in arc length (same units as x,y). If provided, overrides n_samples.
    n_samples : int, optional
        Number of uniformly spaced samples (default 200). Ignored if spacing is given.
    method : {"linear","cubic"}
        Interpolation method. "cubic" uses SciPy if available; falls back to linear.
    smooth_sigma : float
        Optional Gaussian smoothing (in *samples* along the index), e.g. 1–2 to tame pixel jaggies.
        0 disables smoothing.
    closed : bool
        If True, treat curve as closed (include final segment back to the start during length calc).

    Returns
    -------
    s_uniform : (M,) np.ndarray
        Uniform arc-length parameter from 0 to total length L.
    x_uniform, y_uniform : (M,) np.ndarray
        Resampled coordinates at uniform arc-length.
    s_cum : (N,) np.ndarray
        Cumulative arc-length of the (possibly smoothed) input polyline.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Drop NaNs and exact duplicates
    good = ~(np.isnan(x) | np.isnan(y))
    x, y = x[good], y[good]
    if x.size < 2:
        raise ValueError("Need at least two valid points.")

    keep = np.ones_like(x, dtype=bool)
    keep[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    x, y = x[keep], y[keep]
    if x.size < 2:
        raise ValueError("All points are duplicates.")

    # If closed, append the first point at the end if it's not already there
    if closed:
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.concatenate([x, x[:1]])
            y = np.concatenate([y, y[:1]])

    # Optional Gaussian smoothing along the *sequence index*
    if smooth_sigma and smooth_sigma > 0:
        r = int(np.ceil(4 * smooth_sigma))
        t = np.arange(-r, r + 1, dtype=float)
        g = np.exp(-(t**2) / (2 * smooth_sigma**2))
        g /= g.sum()
        mode = "wrap" if closed else "reflect"
        x = np.convolve(np.pad(x, (r, r), mode), g, mode="valid")
        y = np.convolve(np.pad(y, (r, r), mode), g, mode="valid")

    # Cumulative arc length
    ds = np.hypot(np.diff(x), np.diff(y))
    s_cum = np.concatenate([[0.0], np.cumsum(ds)])
    L = float(s_cum[-1])

    # Build uniform arc-length grid
    if spacing is not None:
        if spacing <= 0:
            raise ValueError("spacing must be positive.")
        s_uniform = np.arange(0.0, L, spacing)
        if s_uniform.size == 0 or s_uniform[-1] < L:
            s_uniform = np.append(s_uniform, L)
    else:
        n = max(2, int(n_samples))
        s_uniform = np.linspace(0.0, L, n)

    # Interpolation helpers
    def interp_1d(s, v, su, method):
        if method == "linear":
            return np.interp(su, s, v)
        elif method == "cubic":
            try:
                from scipy.interpolate import CubicSpline
            except Exception:
                # Fallback to linear if SciPy isn't available
                return np.interp(su, s, v)
            # Use natural boundary conditions
            cs = CubicSpline(s, v, bc_type="natural")
            return cs(su)
        else:
            raise ValueError("method must be 'linear' or 'cubic'.")

    x_uniform = interp_1d(s_cum, x, s_uniform, method)
    y_uniform = interp_1d(s_cum, y, s_uniform, method)

    return s_uniform, x_uniform, y_uniform, s_cum



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
    # print("Curve ratio: {}".format(ratio))  

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



def smooth_with_univariate_spline(pixel_curve, smoothing_factor=None, num_points=400):
    """
    Smooth x and y separately as functions of arc length
    """

    # size = int(len(pixel_curve[:, 0]) * 0.25)
    size = num_points

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



def smooth_with_regularization(pixel_curve, smooth=0.1):
    """
    Smooth both x and y with curvature regularization
    """
    # size = int(len(pixel_curve[:, 0]) * 0.25)
    size = pixel_curve.shape[0]
    
    # Calculate arc length
    dx = np.diff(pixel_curve[:, 0])
    dy = np.diff(pixel_curve[:, 1])
    s = np.concatenate(([0], np.cumsum(np.sqrt(dx**2 + dy**2))))
    
    x_orig = pixel_curve[:, 0]
    y_orig = pixel_curve[:, 1]
    
    # Fit smoothing splines to both coordinates
    # Using different smoothing factors for demonstration
    spline_x = UnivariateSpline(s, x_orig, s=smooth * len(s))
    spline_y = UnivariateSpline(s, y_orig, s=smooth * len(s))
    
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
    


def average_min_distance(curve1, curve2):
    """
    Computes the average distance from each point in curve1
    to the nearest point in curve2.

    Parameters:
        curve1 (np.ndarray): Array of shape (n, 2) with (x, y) points.
        curve2 (np.ndarray): Array of shape (m, 2) with (x, y) points.

    Returns:
        float: The average of the minimum distances.
    """
    # Build a KD-tree for efficient nearest-neighbor search
    tree = cKDTree(curve2)
    
    # Query the nearest neighbor in curve2 for each point in curve1
    distances, _ = tree.query(curve1, k=1)
    
    # Compute and return the mean distance
    return np.mean(distances)



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


# measure_neuron_tree('neuron_config_2.yaml', 'fish01_2.CNG.tif')


start_time = time.perf_counter()

# test_curve_interpolation("/Users/zianfanti/IIMAS/images_databases/curves", "images", "coordinates_curves.txt")
test_curve_interpolation('/Users/zianfanti/IIMAS/images_databases/curves', "images", "coordinates_curves.txt", rate=0.50)


# for r in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
#     print("---- Rate: {} ----".format(r))
#     test_curve_interpolation('/Users/zianfanti/IIMAS/images_databases/curves', "images", "coordinates_curves.txt", rate=r)

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.6f} seconds") 

